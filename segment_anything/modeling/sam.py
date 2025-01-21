# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .probabilistic_model import Probabilistic_model
class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.batchsize = self.image_encoder.batchsize
        self.latent_dim = self.image_encoder.latent_dim
        self.in_chans = 3
        self.Probabilistic_model = Probabilistic_model(self.in_chans, self.latent_dim)
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # @torch.no_grad()
    def forward(
    self,
    batched_input: List[Dict[str, Any]],
    multimask_output: bool,
) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        masks_segm = torch.stack([self.preprocess_labels(x["masks"]) for x in batched_input], dim=0)
        sample_z, kl_div = self.Probabilistic_model(input_images, masks_segm)

        all_points = []
        all_boxes = []
        all_masks = []
        for image_record in batched_input:
            point_coords = image_record.get("point_coords", None)
            point_labels = image_record.get("point_labels", None)
            if point_coords is not None and point_labels is not None:
                points = (point_coords, point_labels)
            else:
                points = None
            all_points.append(points)
            boxes = image_record.get("boxes", None)
            all_boxes.append(boxes)
            masks = image_record.get("mask_inputs", None)
            all_masks.append(masks)
        sparse_embeddings_list = []
        dense_embeddings_list = []
        for points, boxes, masks in zip(all_points, all_boxes, all_masks):
            sparse_emb, dense_emb = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )
            sparse_embeddings_list.append(sparse_emb)
            dense_embeddings_list.append(dense_emb)

        sparse_embeddings = torch.stack(sparse_embeddings_list, dim=0)
        dense_embeddings = torch.stack(dense_embeddings_list, dim=0)

        mean_pooled_prompt = torch.mean(sparse_embeddings, dim=1)
        max_pooled_prompt, _ = torch.max(sparse_embeddings, dim=1)
        prompt_embeddings = torch.cat([max_pooled_prompt, mean_pooled_prompt], dim=-1)

        image_embeddings = self.image_encoder(input_images, sample_z, prompt_embeddings)

        outputs = []
        for idx, (image_record, curr_embedding) in enumerate(zip(batched_input, image_embeddings)):
            curr_sparse_embeddings = sparse_embeddings[idx]
            curr_dense_embeddings = dense_embeddings[idx]

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=curr_sparse_embeddings,
                dense_prompt_embeddings=curr_dense_embeddings,
                multimask_output=multimask_output,
            )

            # 后处理生成的 masks
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            })

        return outputs, kl_div

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std
        if x.shape[0]!=3 :
            x = x.repeat(3, 1, 1)
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    def preprocess_labels(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x