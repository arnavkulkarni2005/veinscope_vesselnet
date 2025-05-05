import torch

import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, input_channels=3, n_classes=1):
        super().__init__()
        # Encoder blocks
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Upsample layer (scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder blocks
        self.dec5 = self.deconv_block(512, 512)
        self.dec4 = self.deconv_block(512 + 512, 256)
        self.dec3 = self.deconv_block(256 + 256, 128)
        self.dec2 = self.deconv_block(128 + 128, 64)
        # Final conv: concatenation of dec2 output and enc1 output => 64+64=128
        self.dec1 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def deconv_block(self, in_ch, out_ch):
        # Using convolutional decoder blocks
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder forward
        x1 = self.enc1(x)           # [B,64,H,W]
        x2 = self.enc2(self.pool(x1))  # [B,128,H/2,W/2]
        x3 = self.enc3(self.pool(x2))  # [B,256,H/4,W/4]
        x4 = self.enc4(self.pool(x3))  # [B,512,H/8,W/8]
        x5 = self.enc5(self.pool(x4))  # [B,512,H/16,W/16]

        d5 = self.dec5(x5)  

        d4 = self.upsample(d5)  # [B,256,H/4,W/4]
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.upsample(d4)  # [B,128,H/2,W/2]
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.upsample(d3)  # [B,64,H,W]
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        # Final upsample and conv
        d1 = self.upsample(d2)
        out = self.dec1(torch.cat([d1, x1], dim=1))  # [B,n_classes,H,W]
        # Note: for final, typically you either skip x or not. Here we skip x1 again:

        return self.sigmoid(out) 

def get_masking(original, mask):
    """
    Combines the original image with the mask to generate a masked image.

    Args:
        original (torch.Tensor): The original image tensor of shape (N, C, H, W).
        mask (torch.Tensor): The mask tensor of shape (N, 1, H, W).

    Returns:
        torch.Tensor: The masked image tensor of shape (N, C, H, W).
    """
    # Ensure the mask has the same number of channels as the original image
    if mask.shape[1] == 1:
        mask = mask.repeat(1, original.shape[1], 1, 1)
    
    # Element-wise multiplication to apply the mask
    masked_image = original * mask
    return masked_image

# import cv2
# orig_image_path = '/home/teaching/DL_Hack/filtered_dataset/images/1L_l_1.jpg'
# masked_image_path = '/home/teaching/DL_Hack/filtered_dataset/masks/1L_l_1_vessels.png'
# original = cv2.imread(orig_image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
# mask = cv2.imread(masked_image_path, cv2.IMREAD_GRAYSCALE)    # Load as grayscale

# # Convert images to tensors and add batch and channel dimensions
# original = torch.tensor(original, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
# mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
# masked_image = get_masking(original, mask)
# masked_image = masked_image.squeeze(0)  # Remove the batch dimension
# masked_image = masked_image.permute(1, 2, 0)  # Change to HWC format for visualization
# masked_image = (masked_image * 255).byte().numpy()  # Convert to uint8 for visualization
# import matplotlib.pyplot as plt

# plt.imshow(masked_image, cmap='gray')
# plt.title('Masked Image')
# plt.axis('off')
# plt.show()

