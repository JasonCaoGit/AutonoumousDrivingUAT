import os
import cv2
import csv
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np

# ----- (Optional) Elastic Deformation Function -----
# (Not used for raw images in this script)
def elastic_deformation(image, alpha=1.0, sigma=8.0):
    """Apply elastic deformation to an image (if needed). Not used here for raw images."""
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

# ----- Boundary Noise Function -----
def add_boundary_noise(mask, noise_probability=0.05, noise_radius=3):
    """Add noise to the boundaries of a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    noisy_mask = mask.copy()
    for contour in contours:
        for point in contour:
            x, y = point[0]
            y_min = max(0, y - noise_radius)
            y_max = min(mask.shape[0], y + noise_radius + 1)
            x_min = max(0, x - noise_radius)
            x_max = min(mask.shape[1], x + noise_radius + 1)
            for i in range(y_min, y_max):
                for j in range(x_min, x_max):
                    if random.random() < noise_probability:
                        noisy_mask[i, j] = 255 - noisy_mask[i, j]
    return noisy_mask

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
raw_img_dir = "train"            # Folder containing raw images
gt_mask_dir = "train_labels"       # Folder containing ground truth masks (filenames: <base>_L.png)
output_root = "processed_dataset"  # Root folder for processed outputs
if not os.path.exists(output_root):
    os.makedirs(output_root)

# ----- Configure Noise Parameters -----
noise_probability = 0.05  # Boundary noise probability
noise_radius = 3          # Noise radius

# ----- Process Each Raw Image -----
# For each raw image in "train", there is a corresponding mask in "train_labels" named as <base>_L.png.
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
            cropped_raw = raw_img[y:y+h, x:x+w]
            cropped_mask = binary_mask[y:y+h, x:x+w]

            # Create an output folder for this instance.
            # Folder name format: "<class_name>_<base_name>_<instance_counter>"
            instance_folder = os.path.join(output_root, f"{class_name}_{base_name}_{instance_counter}")
            if not os.path.exists(instance_folder):
                os.makedirs(instance_folder)

            # Save the cropped raw image with original naming: e.g., "car_1.png"
            cropped_raw_filename = f"{class_name}_{base_name}_{instance_counter}.png"
            cropped_raw_path = os.path.join(instance_folder, cropped_raw_filename)
            cv2.imwrite(cropped_raw_path, cropped_raw)

            # Save the cropped original binary mask as e.g., "car_1_seg_1.png"
            orig_mask_filename = f"{class_name}_{base_name}_{instance_counter}_seg_1.png"
            orig_mask_path = os.path.join(instance_folder, orig_mask_filename)
            cv2.imwrite(orig_mask_path, cropped_mask)

            # Create a noisy version of the cropped binary mask
            noisy_mask = add_boundary_noise(cropped_mask, noise_probability, noise_radius)
            noisy_mask_filename = f"{class_name}_{base_name}_{instance_counter}_seg_2.png"
            noisy_mask_path = os.path.join(instance_folder, noisy_mask_filename)
            cv2.imwrite(noisy_mask_path, noisy_mask)

            instance_counter += 1

print("\nDone! Processed dataset has been created in the folder:", output_root)
