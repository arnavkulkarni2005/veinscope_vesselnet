# import os
# import shutil

# # Set your base SBVPI dataset directory
# base_dir = '/home/teaching/DL_Hack/SBVPI'
# output_dir = 'filtered_dataset'
# images_dir = os.path.join(output_dir, 'images')
# masks_dir = os.path.join(output_dir, 'masks')
# cropping_masks_dir = os.path.join(output_dir, 'cropping_masks')

# # Create the output directories if they don't exist
# os.makedirs(images_dir, exist_ok=True)
# os.makedirs(masks_dir, exist_ok=True)
# os.makedirs(cropping_masks_dir, exist_ok=True)

# # Iterate through folders 1 to 55
# for i in range(1, 56):
#     folder_path = os.path.join(base_dir, str(i))
#     if not os.path.isdir(folder_path):
#         continue

#     for file in os.listdir(folder_path):
#         if file.lower().endswith('.jpg'):
#             base_name = os.path.splitext(file)[0]  # e.g., x
#             original_path = os.path.join(folder_path, file)

#             # Check for corresponding vessel mask
#             vessels_filename = f"{base_name}_vessels.png"
#             vessels_path = os.path.join(folder_path, vessels_filename)

#             # Check for corresponding periocular cropping mask
#             crop_filename = f"{base_name}_sclera.png"
#             crop_filename_2 = f"{base_name}_eyelashes.png"
#             crop_path = os.path.join(folder_path, crop_filename)
#             crop_path_2 = os.path.join(folder_path, crop_filename_2)

#             # Only copy if vessel mask exists (as before)
#             if os.path.isfile(vessels_path) and os.path.isfile(crop_path) and os.path.isfile(crop_path_2):
#                 shutil.copy2(original_path, os.path.join(images_dir, file))
#                 shutil.copy2(vessels_path, os.path.join(masks_dir, vessels_filename))

#                 # Copy cropping mask if it exists
#                 shutil.copy2(crop_path, os.path.join(cropping_masks_dir, crop_filename))
#                 shutil.copy2(crop_path_2, os.path.join(cropping_masks_dir, crop_filename_2))

# print("Filtering and copying complete.")
import os
import shutil
from pathlib import Path

# Configuration
BASE_DIR       = Path('/home/teaching/DL_Hack/SBVPI')
OUTPUT_DIR     = Path('filtered_dataset')
IMG_EXTS       = {'.jpg', '.jpeg', '.png'}
CROP_SUFFIXES  = ['_sclera.png', '_eyelashes.png']

# Prepare output folders
IMG_OUT        = OUTPUT_DIR / 'images'
VESSEL_OUT     = OUTPUT_DIR / 'masks'
CROP_OUT       = OUTPUT_DIR / 'cropping_masks'

for folder in (IMG_OUT, VESSEL_OUT, CROP_OUT):
    folder.mkdir(parents=True, exist_ok=True)

missing_log = []

# Walk through each numbered subfolder in BASE_DIR
for sub in BASE_DIR.iterdir():
    if not sub.is_dir() or not sub.name.isdigit():
        continue

    # For each potential image file
    for img_file in sub.iterdir():
        ext = img_file.suffix.lower()
        if ext not in IMG_EXTS:
            continue

        stem = img_file.stem  # e.g. 'x'
        # Construct expected mask paths
        vessel_mask = sub / f"{stem}_vessels.png"
        crop_masks  = [sub / (stem + suffix) for suffix in CROP_SUFFIXES]

        # Check that *all* required files exist
        all_exist = vessel_mask.exists() and all(m.exists() for m in crop_masks)
        if not all_exist:
            missing_log.append(str(sub / img_file.name))
            continue

        # Copy image
        shutil.copy2(img_file, IMG_OUT / img_file.name)
        # Copy vessel mask
        shutil.copy2(vessel_mask, VESSEL_OUT / vessel_mask.name)
        # Copy cropping masks
        for m in crop_masks:
            shutil.copy2(m, CROP_OUT / m.name)

# Report any missing cases
if missing_log:
    print(f"Skipped {len(missing_log)} images due to missing masks. Sample:")
    for p in missing_log[:5]:
        print("  •", p)
else:
    print("All images had corresponding masks—copying complete!")
