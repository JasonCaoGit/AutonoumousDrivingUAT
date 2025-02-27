#!/usr/bin/env python3
import os

def rename_segmentation_files(root_dir):
    """
    Walk through all subfolders in 'root_dir' and rename files that
    contain '_seg_cup_' by removing the "cup" portion.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file is a PNG and contains the pattern "_seg_cup_"
            if filename.endswith('.png') and "_seg_cup_" in filename:
                new_filename = filename.replace("_seg_cup_", "_seg_")
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                print(f"Renaming: {old_path} -> {new_path}")
                os.rename(old_path, new_path)
            # (Optional) Uncomment the following block if you want to delete disc segmentation files:
            # elif filename.endswith('.png') and "_seg_disc_" in filename:
            #     old_path = os.path.join(dirpath, filename)
            #     print(f"Deleting disc segmentation file: {old_path}")
            #     os.remove(old_path)

if __name__ == "__main__":
    # Change this path to point to your REFUGE dataset folder.
    # The script will walk the directory structure under this folder.
    refuge_folder = "C:/Users/jason/OneDrive/Desktop/QMind/UASAM 2/Uncertainty-Aware-Adapter/dataset/REFUGE"

    rename_segmentation_files(refuge_folder)
