import os
import cv2
import csv
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import imgaug.augmenters as iaa
import albumentations as A

# Fix numpy deprecated types
np.complex = np.complex_
np.bool = bool

# ----- Weather Effect Functions -----
def generate_rain_image(image):
    """Applies a moderate rain effect using imgaug."""
    aug = iaa.Rain(
        drop_size=(0.1, 0.15),  # Smaller drops
        speed=(0.1, 0.3)  # Slower rain
    )
    return aug.augment_image(image)

# ----- Elastic Deformation Function -----
def elastic_deformation(image, alpha=1.0, sigma=8.0):
    """Apply elastic deformation to an image."""
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x_map = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    y_map = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
    if len(shape) == 3:
        deformed = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    else:
        deformed = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return deformed, (x_map, y_map)

# ----- Weather-inspired deformation parameters -----
deformation_params = [
    {"alpha": 20.0, "sigma": 15.0},  # Fog-like: higher sigma for smoother, blurred boundaries
    {"alpha": 25.0, "sigma": 4.0},  # Rain-like: sharper, more localized changes
    {"alpha": 30.0, "sigma": 7.0}  # Snow-like: medium smoothness with stronger distortion
]

# ----- Load Class Dictionary and Filter Target Classes -----
target_classes = {"bus", "pedestrian", "car"}
rgb_dict = {}
print("Loading class dictionary...")
with open('class_dict.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        label_name = row[0].strip().lower()
        if label_name in target_classes:
            r, g, b = int(row[1]), int(row[2]), int(row[3])
            rgb_dict[label_name] = [r, g, b]
print(f"Found {len(rgb_dict)} target classes: {list(rgb_dict.keys())}")

# ----- Set Input and Output Directories -----
raw_img_dir = "train"  # Folder containing raw images
gt_mask_dir = "train_labels"  # Folder containing ground truth masks (filenames: <base>_L.png)
output_root = "processed_dataset"  # Root folder for processed outputs
if not os.path.exists(output_root):
    os.makedirs(output_root)

# ----- Process Each Raw Image -----
raw_files = [f for f in os.listdir(raw_img_dir) if f.endswith('.png')]
total_files = len(raw_files)
print(f"\nProcessing {total_files} images...")

for file_name in tqdm(raw_files, desc="Processing images"):
    raw_path = os.path.join(raw_img_dir, file_name)
    base_name = os.path.splitext(file_name)[0]
    gt_path = os.path.join(gt_mask_dir, base_name + "_L.png")

    # Read raw image and corresponding ground truth mask
    raw_img = cv2.imread(raw_path)
    gt_img = cv2.imread(gt_path)
    if raw_img is None:
        print(f"Warning: Cannot read raw image {raw_path}. Skipping.")
        continue
    if gt_img is None:
        print(f"Warning: Cannot read ground truth mask {gt_path}. Skipping.")
        continue

    # For each target class, extract its binary mask from the GT image
    for class_name, rgb in rgb_dict.items():
        # Create binary mask for this class (compare against [B, G, R])
        binary_mask = np.all(gt_img == [rgb[2], rgb[1], rgb[0]], axis=2).astype(np.uint8) * 255
        if not np.any(binary_mask):
            continue  # Skip if this class is not present

        # Find contours for separate instances
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        instance_counter = 1
        for cnt in contours:
            # Compute bounding box for the contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Optionally, ignore very small regions
            if w * h < 100:  # Adjust threshold as needed
                continue

            # Crop the raw image and the binary mask to the bounding box
            cropped_raw = raw_img[y:y + h, x:x + w]
            cropped_mask = binary_mask[y:y + h, x:x + w]

            # Create an output folder for this instance
            instance_folder = os.path.join(output_root, f"{class_name}_{base_name}_{instance_counter}")
            if not os.path.exists(instance_folder):
                os.makedirs(instance_folder)

            # Apply rain effect to the cropped raw image
            # Convert BGR to RGB for imgaug
            cropped_raw_rgb = cv2.cvtColor(cropped_raw, cv2.COLOR_BGR2RGB)

            # Process for rain effect
            rain_img = generate_rain_image(cropped_raw_rgb.copy())

            # Convert back to BGR for saving
            rain_img_bgr = cv2.cvtColor(rain_img, cv2.COLOR_RGB2BGR)

            # Save the rain-affected image with the original naming
            orig_img_filename = f"{class_name}_{base_name}_{instance_counter}.png"
            orig_img_path = os.path.join(instance_folder, orig_img_filename)
            cv2.imwrite(orig_img_path, rain_img_bgr)

            # Save the cropped original binary mask as e.g., "car_1_seg_1.png"
            orig_mask_filename = f"{class_name}_{base_name}_{instance_counter}_seg_1.png"
            orig_mask_path = os.path.join(instance_folder, orig_mask_filename)
            cv2.imwrite(orig_mask_path, cropped_mask)

            # Generate three different deformed versions of the mask
            for i, params in enumerate(deformation_params, start=2):
                # Apply elastic deformation with parameters matching weather effects
                deformed_mask, _ = elastic_deformation(cropped_mask,
                                                       alpha=params["alpha"],
                                                       sigma=params["sigma"])

                # Ensure the deformed mask stays binary
                deformed_mask = (deformed_mask > 127).astype(np.uint8) * 255

                # Save the deformed mask
                deformed_mask_filename = f"{class_name}_{base_name}_{instance_counter}_seg_{i}.png"
                deformed_mask_path = os.path.join(instance_folder, deformed_mask_filename)
                cv2.imwrite(deformed_mask_path, deformed_mask)

            instance_counter += 1

print("\nDone! Processed dataset has been created in the folder:", output_root)