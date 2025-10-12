# prostatex-t2-prostate-train-attention.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the new AttentionUNet instead of regular UNet
from unet.unet_attention import AttentionUNet
from prostate_dataset import ProstateMRIDataset


def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

# ✅ 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ✅ 数据路径设置
IMAGE_DIR = './unified_dataset_v2/prostate_t2/images_renamed'
MASK_DIR = './unified_dataset_v2/prostate_t2/masks_renamed'
TRAIN_SPLIT_TXT = './unified_dataset_v2/prostate_t2/splits/train.txt'
VAL_SPLIT_TXT = './unified_dataset_v2/prostate_t2/splits/val.txt'
CHECKPOINT_DIR = './Pytorch-UNet-master/Pytorch-UNet-master/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ✅ 加载数据集
# For 256x320 images, you can choose:
# target_size=(256, 320)  # Keep original aspect ratio
# target_size=(256, 256)  # Square (current default)
# target_size=(320, 320)  # Square with larger size
# target_size=None        # No resizing (use original size)

train_dataset = ProstateMRIDataset(
    IMAGE_DIR, MASK_DIR, TRAIN_SPLIT_TXT, 
    target_size=(256, 320),  # Keep original aspect ratio
    search_subfolders=True   # Enable subfolder search
)
val_dataset = ProstateMRIDataset(
    IMAGE_DIR, MASK_DIR, VAL_SPLIT_TXT,
    target_size=(256, 320),  # Keep original aspect ratio  
    search_subfolders=True   # Enable subfolder search
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ✅ 初始化AttentionUNet模型
model = AttentionUNet(
    n_channels=1,
    n_classes=1,
    bilinear=True,
    base_k_max=32,  # Base k_max, will be dynamically adjusted
    use_attention=True,
    attention_layers=['down3', 'down4']  # All encoder + bottleneck
).to(device)

# Print model information
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📦 AttentionUNet Parameters: Total = {total_params:,}, Trainable = {trainable_params:,}")

# Count attention modules
attention_modules = sum(1 for name, _ in model.named_modules() if 'attention' in name)
print(f"🔍 Number of attention modules: {attention_modules}")

# ✅ 损失函数与优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ✅ 训练参数
num_epochs = 100
best_val_loss = float('inf')
train_losses, val_losses = [], []

# ✅ 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"🟢 [Epoch {epoch+1}/{num_epochs}]")

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs)
            dice_total += dice_score(preds, masks) * images.size(0)
            iou_total += iou_score(preds, masks) * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_dice = dice_total / len(val_loader.dataset)
    val_iou = iou_total / len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"📊 Epoch {epoch+1}: "
          f"Train Loss = {epoch_loss:.4f}, "
          f"Val Loss = {val_loss:.4f}, "
          f"Val Dice = {val_dice:.4f}, "
          f"Val IoU = {val_iou:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'model_config': {
                'n_channels': 1,
                'n_classes': 1,
                'bilinear': True,
                'base_k_max': 32,
                'use_attention': True,
                # 'attention_layers': ['inc', 'down1', 'down2', 'down3', 'down4']
                'attention_layers': ['down3', 'down4']
            }
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'attention_unet_best.pth'))
        print("✅ Saved best model")

# ✅ 保存最终模型
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'attention_unet_final.pth'))
print("🎉 Training completed and final model saved!")

# ✅ 可视化损失曲线
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('AttentionUNet Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CHECKPOINT_DIR, 'attention_loss_curve.png'))
plt.show()