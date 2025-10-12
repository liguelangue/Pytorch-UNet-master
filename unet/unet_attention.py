# unet/unet_attention.py
"""
U-Net with KNN Attention in encoder and bottleneck
"""
import torch
import torch.nn as nn
from .unet_parts import DoubleConv, Up, OutConv
from .attention_modules import KNNAttention, get_dynamic_k_max


class AttentionDoubleConv(nn.Module):
    """DoubleConv block with optional KNN Attention"""
    def __init__(self, in_channels, out_channels, mid_channels=None, 
                 use_attention=True, base_k_max=32):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Original double convolution
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Optional KNN Attention
        self.use_attention = use_attention
        self.base_k_max = base_k_max
        if use_attention:
            # Initialize with base_k_max, will be dynamically adjusted in forward
            self.attention = KNNAttention(
                out_channels, 
                k_max=base_k_max,
                k_min=4,
                adaptive=True
            )
    
    def forward(self, x):
        x = self.double_conv(x)
        
        if self.use_attention:
            # Dynamically adjust k_max based on current feature map size
            _, _, h, w = x.shape
            dynamic_k = get_dynamic_k_max(h, w, self.base_k_max)
            
            # Update k_max if different from current
            if self.attention.k_max != dynamic_k:
                self.attention.k_max = dynamic_k
            
            x = self.attention(x)
        
        return x


class AttentionDown(nn.Module):
    """Downscaling with maxpool then attention-enhanced double conv"""
    def __init__(self, in_channels, out_channels, use_attention=True, base_k_max=32):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = AttentionDoubleConv(in_channels, out_channels, 
                                       use_attention=use_attention, 
                                       base_k_max=base_k_max)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """U-Net with KNN Attention in all encoder layers and bottleneck"""
    def __init__(self, n_channels, n_classes, bilinear=False, 
                 base_k_max=32, use_attention=True,
                 attention_layers=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_k_max = base_k_max
        
        # Default: attention in all encoder layers and bottleneck
        if attention_layers is None:
            attention_layers = ['inc', 'down1', 'down2', 'down3', 'down4']
        self.attention_layers = set(attention_layers)
        
        # Encoder with conditional attention
        self.inc = AttentionDoubleConv(n_channels, 64, 
                                      use_attention='inc' in self.attention_layers,
                                      base_k_max=base_k_max)
        
        self.down1 = AttentionDown(64, 128,
                                  use_attention='down1' in self.attention_layers,
                                  base_k_max=base_k_max)
        
        self.down2 = AttentionDown(128, 256,
                                  use_attention='down2' in self.attention_layers,
                                  base_k_max=base_k_max)
        
        self.down3 = AttentionDown(256, 512,
                                  use_attention='down3' in self.attention_layers,
                                  base_k_max=base_k_max)
        
        factor = 2 if bilinear else 1
        self.down4 = AttentionDown(512, 1024 // factor,
                                  use_attention='down4' in self.attention_layers,
                                  base_k_max=base_k_max)
        
        # Decoder (no attention, using original Up blocks)
        self.up1 = Up(1024, 512 // factor, bilinear)
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
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        # Wrap each module with checkpointing
        import torch.utils.checkpoint as checkpoint
        
        # Note: This is a simplified version. In practice, you'd need to 
        # properly handle the forward functions with checkpoint
        self.inc = checkpoint(self.inc)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.up1 = checkpoint(self.up1)
        self.up2 = checkpoint(self.up2)
        self.up3 = checkpoint(self.up3)
        self.up4 = checkpoint(self.up4)
        self.outc = checkpoint(self.outc)