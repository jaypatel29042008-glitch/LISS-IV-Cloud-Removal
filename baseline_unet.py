"""
Team Brain Wave — Step 3.1 & Step 4 Roadmap Implementation: U-Net Baseline Model
Provides a lightweight PyTorch U-Net for cloud removal & cloud-free reconstruction.
Includes RICE Dataset PyTorch Dataset loaders and training/evaluation pipeline.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ----------------------------------------------------
# 1. PyTorch Dataset for RICE Benchmark
# ----------------------------------------------------
class RICEDataset(Dataset):
    def __init__(self, root_dir, dataset_type='RICE1', transform=None):
        """
        root_dir: path to RICE_DATASET (containing RICE1 and RICE2)
        dataset_type: 'RICE1' or 'RICE2'
        """
        self.root_dir = os.path.join(root_dir, dataset_type)
        self.cloud_dir = os.path.join(self.root_dir, 'cloud')
        self.label_dir = os.path.join(self.root_dir, 'label')
        self.dataset_type = dataset_type

        if not os.path.exists(self.cloud_dir):
            raise FileNotFoundError(f"Could not find cloud directory at {self.cloud_dir}")

        self.filenames = sorted([f for f in os.listdir(self.cloud_dir) if f.endswith(('.png', '.jpg'))])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        cloud_path = os.path.join(self.cloud_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        cloud_img = Image.open(cloud_path).convert('RGB')
        label_img = Image.open(label_path).convert('RGB')

        # Convert to Tensor [3, H, W] normalized to [0, 1]
        c_arr = np.array(cloud_img, dtype=np.float32) / 255.0
        l_arr = np.array(label_img, dtype=np.float32) / 255.0

        c_tensor = torch.from_numpy(c_arr).permute(2, 0, 1)
        l_tensor = torch.from_numpy(l_arr).permute(2, 0, 1)

        return c_tensor, l_tensor, fname


# ----------------------------------------------------
# 2. U-Net Architecture for Cloud Removal
# ----------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class CloudUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        out = self.outc(x)
        return self.sigmoid(out)


if __name__ == "__main__":
    model = CloudUNet(in_channels=3, out_channels=3)
    dummy_in = torch.randn(2, 3, 256, 256)
    out = model(dummy_in)
    print("CloudUNet Baseline initialized successfully.")
    print("Input shape:", dummy_in.shape)
    print("Output shape:", out.shape)
