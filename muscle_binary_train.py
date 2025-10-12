# muscle-binary-train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet  # Implementation from milesial
from muscle_dataset import MuscleMRIDataset


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


def main():
    # ✅ Auto-select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # ✅ Data path settings
    IMAGE_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\converted_dataset_binary\binary_dataset_class_90\train\images'
    MASK_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\converted_dataset_binary\binary_dataset_class_90\train\masks'
    TRAIN_SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\converted_dataset_binary\binary_dataset_class_90\splits\train.txt'
    VAL_SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\converted_dataset_binary\binary_dataset_class_90\splits\val.txt'
    CHECKPOINT_DIR = './checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ✅ Load dataset
    # For 256x320 images, you can choose:
    # target_size=(256, 320)  # Keep original aspect ratio
    target_size=(256, 256)  # Square (current default)

    train_dataset = MuscleMRIDataset(
        IMAGE_DIR, MASK_DIR, TRAIN_SPLIT_TXT, 
        transform=None,
        target_size=target_size,
        search_subfolders=True
    )
    val_dataset = MuscleMRIDataset(
        IMAGE_DIR, MASK_DIR, VAL_SPLIT_TXT,
        transform=None,
        target_size=target_size,
        search_subfolders=True
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # ✅ Initialize model
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 Model Parameters: Total = {total_params:,}, Trainable = {trainable_params:,}")

    # ✅ Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # Increase learning rate for faster convergence
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ✅ Training parameters
    num_epochs = 50  # Reduced from 100 to 50
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience = 3  # Early stopping patience value
    patience_counter = 0  # Early stopping counter

    # ✅ Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"🟢 [Epoch {epoch+1}/{num_epochs}]")

        for images, masks, filenames in pbar:
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

        # Validation phase
        print("Validation started...")
        model.eval()
        val_loss = 0.0
        dice_total = 0.0
        iou_total = 0.0
        with torch.no_grad():
            for images, masks, filenames in val_loader:
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


        # ✅ Save best model and early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset early stopping counter
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'unet_muscle_90_best.pth'))
            print("✅ Saved best model")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")
        
        # ✅ Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'unet_muscle_90_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

    # ✅ Visualize loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve.png'))
    plt.show()

if __name__ == '__main__':
    main()