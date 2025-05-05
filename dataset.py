# import os
# import cv2
# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from albumentations.core.transforms_interface import ImageOnlyTransform

# class FilteredDataset(Dataset):
#     def __init__(self, data_dir, transform=None, indices=None):
#         """
#         Args:
#             data_dir (str): Root directory with subfolders:
#                 - images/         (x.jpg/.jpeg/.png)
#                 - cropping_masks/ (x_sclera.png, x_eyelashes.png)
#             transform (albumentations.Compose, optional): 
#                 transforms applied to image & mask together.
#             indices (list, optional): List of indices to subset the dataset.
#         """
#         self.data_dir  = data_dir
#         self.transform = transform
#         self.ids       = []

#         img_dir  = os.path.join(data_dir, "images")
#         crop_dir = os.path.join(data_dir, "cropping_masks")

#         # scan images/ and pre-filter IDs with both masks present
#         for fname in os.listdir(img_dir):
#             stem, ext = os.path.splitext(fname)
#             if ext.lower() not in (".jpg", ".jpeg", ".png"):
#                 continue

#             sclera_f = os.path.join(crop_dir, f"{stem}_sclera.png")
#             lash_f   = os.path.join(crop_dir, f"{stem}_eyelashes.png")

#             if os.path.isfile(sclera_f) and os.path.isfile(lash_f):
#                 self.ids.append((stem, ext))

#         if not self.ids:
#             raise RuntimeError("No valid images found! Check that your 'images/' and 'cropping_masks/' folders contain matching files.")

#         # Apply indices if provided
#         if indices is not None:
#             self.ids = [self.ids[i] for i in indices if 0 <= i < len(self.ids)]

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         stem, ext = self.ids[idx]

#         # load image
#         img_path = os.path.join(self.data_dir, "images", stem + ext)
#         image_bgr = cv2.imread(img_path)
#         if image_bgr is None:
#             raise FileNotFoundError(f"Unable to read image: {img_path}")
#         image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#         # load and combine masks
#         crop_dir   = os.path.join(self.data_dir, "cropping_masks")
#         sclera_np  = np.array(
#             Image.open(os.path.join(crop_dir, f"{stem}_sclera.png")).convert("L"),
#             dtype=np.float32
#         ) / 255.0
#         eyel_np    = np.array(
#             Image.open(os.path.join(crop_dir, f"{stem}_eyelashes.png")).convert("L"),
#             dtype=np.float32
#         ) / 255.0

#         combined_np = (sclera_np * (1.0 - eyel_np) * 255).astype(np.float32)

#         # apply transforms
#         if self.transform:
#             augmented = self.transform(image=image, mask=combined_np)
#             image    = augmented["image"]
#             combined = augmented["mask"]
#         else:
#             # fallback to tensor conversion
#             image    = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
#             combined = torch.from_numpy(combined_np[None]).float() / 255.0
#         filename = os.path.basename(img_path).split('.')[0]  
        
#         return image.float(), combined.float() , filename

# # Example of constructing a DataLoader
# if __name__ == "__main__":
#     transform = A.Compose([
#         A.Resize(512, 512),
#         ToTensorV2(),
#     ])
#     dataset = FilteredDataset(data_dir="./filtered_dataset", transform=transform)
#     loader  = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
#     imgs, masks = next(iter(loader))
#     print(imgs.shape, masks.shape)

# # Example usage:
# # transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
# # dataset = CustomDataset(data_dir="/path/to/data", transform=transform)
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

class GreenChannelCLAHE(A.ImageOnlyTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), always_apply=False, p=1.0):
        super(GreenChannelCLAHE, self).__init__(always_apply, p)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image, **params):
        image = image.copy()
        
        # Ensure it's in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        green = image[:, :, 1]  # Extract green channel
        green_clahe = self.clahe.apply(green)  # Apply CLAHE

        image[:, :, 1] = green_clahe  # Put back the enhanced green
        return image

    def get_transform_init_args_names(self):
        # ensures repr prints clip_limit & tile_grid_size
        return ("clip_limit", "tile_grid_size")

class OtsuSegmentation:
    """
    Class for applying Otsu's Thresholding for segmentation.
    """
    def __init__(self, resize=(512, 512)):
        self.resize = resize

    def __call__(self, image, mask=None):
        """
        Apply Otsu's method to segment the image.
        Args:
            image (np.ndarray): Input image.
            mask (np.ndarray, optional): Optional mask (not used in Otsu's method).

        Returns:
            (image, mask): Segmented image and mask.
        """
        # Resize the image
        image_resized = cv2.resize(image, self.resize)

        # Convert to grayscale
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert the mask to a float array (for consistency with models)
        otsu_mask = otsu_mask.astype(np.float32) / 255.0

        # Optionally, process the mask (e.g., remove small areas or fill holes)
        # otsu_mask = self.process_mask(otsu_mask)

        return torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0, torch.from_numpy(otsu_mask[None]).float()


class FilteredDataset(Dataset):
    def __init__(self, data_dir, transform=None,indices=None):
        """
        Args:
            data_dir (str): Root directory with subfolders:
                - images/         (x.jpg/.jpeg/.png)
                - cropping_masks/ (x_sclera.png, x_eyelashes.png)
            transform (albumentations.Compose, optional): 
                transforms applied to image & mask together.
        """
        self.data_dir  = data_dir
        self.transform = transform
        self.ids        = []

        img_dir  = os.path.join(data_dir, "images")
        crop_dir = os.path.join(data_dir, "cropping_masks")

        # scan images/ and pre-filter IDs with both masks present
        for fname in os.listdir(img_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            sclera_f = os.path.join(crop_dir, f"{stem}_sclera.png")
            lash_f   = os.path.join(crop_dir, f"{stem}_eyelashes.png")

            if os.path.isfile(sclera_f) and os.path.isfile(lash_f):
                self.ids.append((stem, ext))

        if not self.ids:
            raise RuntimeError("No valid images found! Check that your 'images/' and 'cropping_masks/' folders contain matching files.")
        # Apply indices if provided
        if indices is not None:
            self.ids = [self.ids[i] for i in indices if 0 <= i < len(self.ids)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem, ext = self.ids[idx]

        # load image
        img_path = os.path.join(self.data_dir, "images", stem + ext)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # load and combine masks (for reference)
        crop_dir = os.path.join(self.data_dir, "cropping_masks")
        sclera_np = np.array(
            Image.open(os.path.join(crop_dir, f"{stem}_sclera.png")).convert("L"),
            dtype=np.float32
        ) / 255.0
        eyel_np = np.array(
            Image.open(os.path.join(crop_dir, f"{stem}_eyelashes.png")).convert("L"),
            dtype=np.float32
        ) / 255.0

        # Combine the sclera and eyelash masks into one binary mask
        combined_np = (sclera_np * (1.0 - eyel_np)).astype(np.float32)  # Binary mask (0 or 1)

        # Ensure combined mask is in the range [0, 1]
        combined_np = np.clip(combined_np, 0, 1)

        # apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=combined_np)
            image = augmented["image"]
            combined = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            combined = torch.from_numpy(combined_np[None]).float()  # [None] to add channel dimension

        filename = os.path.basename(img_path).split('.')[0]

        return image.float(), combined.float(), filename


# Example of constructing a DataLoader
# if _name_ == "_main_":
#     otsu_augmentation = OtsuSegmentation(resize=(512, 512))  # Otsu's method for segmentation
#     dataset = FilteredDataset(data_dir="/home/teaching/dl_hack_arnavk_copy/attention_unet/filtered_dataset", transform=otsu_augmentation)
#     loader  = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

#     imgs, masks, filenames = next(iter(loader))
#     print(imgs.shape, masks.shape)

class VesselsSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.masked_images_dir = os.path.join(data_dir, "masked_images")
        self.vessels_dir = os.path.join(data_dir, "masks")
        self.transform = transform

        # Find all valid pairs
        self.valid_pairs = []
        for filename in os.listdir(self.masked_images_dir):
            if filename.endswith("_masked_image.png"):
                id_part = filename.replace("_masked_image.png", "")
                vessels_path = os.path.join(self.vessels_dir, f"{id_part}_vessels.png")
                masked_image_path = os.path.join(self.masked_images_dir, filename)
                if os.path.exists(vessels_path):
                    self.valid_pairs.append((masked_image_path, vessels_path))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.valid_pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask[None]).float()  # [None] to add channel dimension

        filename = os.path.basename(image_path).split('.')[0]

        return image.float(), mask.float(), filename
