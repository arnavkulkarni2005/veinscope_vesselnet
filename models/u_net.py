import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------- UNet Architecture ---------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Bottleneck(nn.Module):
    """Additional conv block at the network's bottom"""
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Bottleneck
        self.bottleneck = Bottleneck(1024 // factor, 1024)

        # Upsampling path
        self.up1 = Up(1024 + (1024 // factor), 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# --------------------- Dice Loss ---------------------

# def make_one_hot(input, num_classes):
#     """Convert class index tensor to one hot encoding tensor.
#     Args:
#          input: A tensor of shape [N, 1, *]
#          num_classes: An int of number of class
#     Returns:
#         A tensor of shape [N, num_classes, *]
#     """
#     shape = np.array(input.shape)
#     shape[1] = num_classes
#     shape = tuple(shape)
#     result = torch.zeros(shape)
#     result = result.scatter_(1, input.cpu().long(), 1)

#     return result


# class BinaryDiceLoss(nn.Module):
#     """Dice loss of binary class
#     Args:
#         smooth: A float number to smooth loss, and avoid NaN error, default: 1
#         p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
#         predict: A tensor of shape [N, *]
#         target: A tensor of shape same with predict
#     Returns:
#         Loss tensor according to arg reduction
#     Raise:
#         Exception if unexpected reduction
#     """
#     def __init__(self, smooth=1, p=2):
#         super(BinaryDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p

#     def forward(self, predict, target):
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#         predict = predict.contiguous().view(predict.shape[0], -1)
#         target = target.contiguous().view(target.shape[0], -1)

#         num = torch.sum(torch.mul(predict, target))*2 + self.smooth
#         den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

#         dice = num / den
#         loss = 1 - dice
#         return loss

# class DiceLoss(nn.Module):
#     """Dice loss, need one hot encode input
#     Args:
#         weight: An array of shape [num_classes,]
#         ignore_index: class index to ignore
#         predict: A tensor of shape [N, C, *]
#         target: A tensor of same shape with predict
#         other args pass to BinaryDiceLoss
#     Return:
#         same as BinaryDiceLoss
#     """
#     def __init__(self, weight=None, ignore_index=None, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#         self.weight = weight
#         self.ignore_index = ignore_index

#     def forward(self, predict, target):
#         if target.shape[1] != predict.shape[1]:
#             target = make_one_hot(target, predict.shape[1])
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         dice = BinaryDiceLoss(**self.kwargs)
#         total_loss = 0
#         predict = F.softmax(predict, dim=1)

#         for i in range(target.shape[1]):
#             if i != self.ignore_index:
#                 dice_loss = dice(predict[:, i], target[:, i])
#                 if self.weight is not None:
#                     assert self.weight.shape[0] == target.shape[1], \
#                         'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
#                     dice_loss *= self.weights[i]
#                 total_loss += dice_loss

#         return total_loss/target.shape[1]

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / \
                     (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice_score

        # Total loss with graph preserved
        return dice_loss

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)

        # Use sigmoid on logits without detaching from graph
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / \
                     (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice_score

        # Total loss with graph preserved
        return bce_loss + dice_loss
from torch.nn import init

# Deformed Non-Local Block
class DNLBlock(nn.Module):
    def __init__(self, in_channels=3):
        super(DNLBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        init.constant_(self.W.weight, 0)
        init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [B, C', N]
        g_x = g_x.permute(0, 2, 1)  # [B, N, C']

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [B, C', N]
        theta_x = theta_x.permute(0, 2, 1)  # [B, N, C']
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C', N]

        f = torch.matmul(theta_x, phi_x)  # [B, N, N]
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # [B, N, C']
        y = y.permute(0, 2, 1).contiguous()  # [B, C', N]
        y = y.view(batch_size, self.inter_channels, H, W)  # [B, C', H, W]
        W_y = self.W(y)
        z = W_y + x
        return z

# Multi-Scale Feature Fusion
class MFF(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MFF, self).__init__()
        self.convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.fusion = nn.Conv2d(len(in_channels_list) * out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        upsampled_features = []
        target_size = features[0].size()[2:]
        for i, feature in enumerate(features):
            x = self.convs[i](feature)
            if x.size()[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(x)
        x = torch.cat(upsampled_features, dim=1)
        x = self.fusion(x)
        return x

# Residual Squeeze and Excitation Pyramid Pooling
class RSEP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RSEP, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv(x)
        b, c, _, _ = residual.size()
        se = self.pool1(residual)
        se = self.conv1x1(se)
        se = self.relu(se)
        se = torch.sigmoid(se)
        out = residual * se
        return out + x

# DNL-Net U-Net
class DNLNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(DNLNetUNet, self).__init__()

        # Further reduced the number of channels and layers
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DNLBlock(16)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DNLBlock(32)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DNLBlock(64)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DNLBlock(128)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DNLBlock(64)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DNLBlock(32)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DNLBlock(16)
        )

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

        # Optional: Reduced number of feature maps in MFF and RSEP
        self.mff = MFF([16, 32, 64], 16)
        self.rsep = RSEP(16, 16)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))

        middle = self.middle(F.max_pool2d(enc3, kernel_size=2))

        dec3 = self.decoder3(F.interpolate(middle, scale_factor=2, mode='bilinear', align_corners=False))
        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False))
        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False))

        fused = self.mff([enc1, enc2, enc3])
        fused = self.rsep(fused)

        out = self.final_conv(fused)
        return out

class CrossHairConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossHairConv2D, self).__init__()
        self.vertical = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.horizontal = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        return self.vertical(x) + self.horizontal(x)

class DeepVesselNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(DeepVesselNet2D, self).__init__()
        self.enc1 = nn.Sequential(
            CrossHairConv2D(in_channels, 32),
            nn.ReLU(inplace=True),
            CrossHairConv2D(32, 32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            CrossHairConv2D(32, 64),
            nn.ReLU(inplace=True),
            CrossHairConv2D(64, 64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            CrossHairConv2D(64, 128),
            nn.ReLU(inplace=True),
            CrossHairConv2D(128, 64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            CrossHairConv2D(128, 64),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            CrossHairConv2D(64, 32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

