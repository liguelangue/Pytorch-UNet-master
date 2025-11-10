# muscle-multi-train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet  # Implementation from milesial
from muscle_dataset_multi import MuscleMRIDatasetMulti


def compute_per_class_metrics(pred, target, num_classes, ignore_index=None):
    """Compute per-class Dice and IoU scores for detailed analysis"""
    per_class_dice = {}
    per_class_iou = {}
    
    # Get hard predictions
    if pred.dim() == 4:  # [B, C, H, W]
        pred_soft = F.softmax(pred, dim=1)
        pred_hard = torch.argmax(pred_soft, dim=1)
    else:
        pred_hard = pred
    
    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue
        
        # Get binary masks for this class
        pred_class = (pred_hard == class_id).float()
        target_class = (target == class_id).float()
        
        # Calculate metrics
        intersection = (pred_class * target_class).sum()
        pred_sum = pred_class.sum()
        target_sum = target_class.sum()
        union = pred_sum + target_sum - intersection
        
        # Dice score
        if target_sum == 0 and pred_sum == 0:
            dice = 1.0
        elif target_sum == 0 or pred_sum == 0:
            dice = 0.0
        else:
            dice = (2.0 * intersection / (pred_sum + target_sum)).item()
        
        # IoU score  
        if union == 0:
            iou = 1.0
        else:
            iou = (intersection / union).item()
        
        per_class_dice[class_id] = dice
        per_class_iou[class_id] = iou
    
    return per_class_dice, per_class_iou


def dice_score_multiclass_correct(pred, target, num_classes, ignore_index=None, smooth=1e-5):
    """
    Calculate Dice score for multi-class segmentation - CORRECTED VERSION
    Properly handles classes that don't exist in the image
    
    Args:
        pred: [B, C, H, W] raw logits
        target: [B, H, W] class indices
        num_classes: number of classes
        ignore_index: class to ignore (default None - include all classes)
    """
    dice_scores = []
    
    # Apply softmax to get probabilities, then argmax for hard predictions
    pred_soft = F.softmax(pred, dim=1)
    pred_hard = torch.argmax(pred_soft, dim=1)  # [B, H, W]
    
    for class_id in range(num_classes):
        # Only skip if explicitly set to ignore
        if ignore_index is not None and class_id == ignore_index:
            continue
        
        # Get binary masks for this class
        pred_class = (pred_hard == class_id).float()
        target_class = (target == class_id).float()
        
        # Count pixels
        pred_sum = pred_class.sum()
        target_sum = target_class.sum()
        intersection = (pred_class * target_class).sum()
        
        # Handle different cases
        if target_sum == 0 and pred_sum == 0:
            # Class doesn't exist and model correctly predicts it doesn't exist
            dice = torch.tensor(1.0, device=pred.device)
        elif target_sum == 0 and pred_sum > 0:
            # Class doesn't exist but model wrongly predicts it exists
            dice = torch.tensor(0.0, device=pred.device)
        elif target_sum > 0 and pred_sum == 0:
            # Class exists but model fails to predict it
            dice = torch.tensor(0.0, device=pred.device)
        else:
            # Normal case: class exists and model makes predictions
            dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        
        dice_scores.append(dice)
    
    if dice_scores:
        return torch.stack(dice_scores).mean()
    else:
        return torch.tensor(1.0, device=pred.device)  # All classes ignored - perfect


def iou_score_multiclass_correct(pred, target, num_classes, ignore_index=None, smooth=1e-5):
    """
    Calculate IoU score for multi-class segmentation - CORRECTED VERSION
    Properly handles classes that don't exist in the image
    """
    iou_scores = []
    
    # Apply softmax and get hard predictions
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    for class_id in range(num_classes):
        # Only skip if explicitly set to ignore
        if ignore_index is not None and class_id == ignore_index:
            continue
        
        # Get binary masks for this class
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()
        
        # Count pixels
        pred_sum = pred_class.sum()
        target_sum = target_class.sum()
        intersection = (pred_class * target_class).sum()
        union = pred_sum + target_sum - intersection
        
        # Handle different cases
        if target_sum == 0 and pred_sum == 0:
            # Class doesn't exist and model correctly predicts it doesn't exist
            iou = torch.tensor(1.0, device=pred.device)
        elif union == 0:
            # Shouldn't happen, but handle edge case
            iou = torch.tensor(1.0, device=pred.device)
        else:
            # Normal IoU calculation
            iou = (intersection + smooth) / (union + smooth)
        
        iou_scores.append(iou)
    
    if iou_scores:
        return torch.stack(iou_scores).mean()
    else:
        return torch.tensor(1.0, device=pred.device)  # All classes ignored - perfect


# Backward compatibility aliases
def dice_score_multi_class(pred, target, num_classes=3, smooth=1e-6, ignore_background=True):
    """
    Calculate Dice score for multi-class segmentation (backward compatibility)
    
    Args:
        pred: [B, C, H, W] raw logits
        target: [B, H, W] class indices
        num_classes: number of classes
        smooth: smoothing factor
        ignore_background: If True, ignore class 0 (background) in metric calculation
    """
    ignore_index = 0 if ignore_background else None
    result = dice_score_multiclass_correct(pred, target, num_classes, ignore_index=ignore_index, smooth=smooth)
    return result.item() if isinstance(result, torch.Tensor) else result


def iou_score_multi_class(pred, target, num_classes=3, smooth=1e-6, ignore_background=True):
    """
    Calculate IoU score for multi-class segmentation (backward compatibility)
    
    Args:
        pred: [B, C, H, W] raw logits
        target: [B, H, W] class indices
        num_classes: number of classes
        smooth: smoothing factor
        ignore_background: If True, ignore class 0 (background) in metric calculation
    """
    ignore_index = 0 if ignore_background else None
    result = iou_score_multiclass_correct(pred, target, num_classes, ignore_index=ignore_index, smooth=smooth)
    return result.item() if isinstance(result, torch.Tensor) else result


def main():
    # ✅ Auto-select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # ✅ Data path settings
    IMAGE_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top3\train\images'
    MASK_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top3\train\masks'
    # TRAIN_SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top3\splits\train.txt'
    # VAL_SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top3\splits\val.txt'
    CHECKPOINT_DIR = './checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ✅ Load dataset
    # For 256x320 images, you can choose:
    # target_size=(256, 320)  # Keep original aspect ratio
    target_size=(256, 256)  # Square (current default)

    train_dataset = MuscleMRIDatasetMulti(
        IMAGE_DIR, MASK_DIR, 
        transform=None,
        target_size=target_size,
        search_subfolders=True,
        num_classes=3
    )
    val_dataset = MuscleMRIDatasetMulti(
        IMAGE_DIR, MASK_DIR,
        transform=None,
        target_size=target_size,
        search_subfolders=True,
        num_classes=3
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # ✅ Initialize model
    model = UNet(n_channels=1, n_classes=3, bilinear=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 Model Parameters: Total = {total_params:,}, Trainable = {trainable_params:,}")

    # ✅ Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # Increase learning rate for faster convergence
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ✅ Training parameters
    num_epochs = 10
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
        per_class_dice_total = {0: 0.0, 1: 0.0, 2: 0.0}
        per_class_iou_total = {0: 0.0, 1: 0.0, 2: 0.0}
        
        with torch.no_grad():
            for images, masks, filenames in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Calculate metrics including all classes (background + muscles)
                dice_total += dice_score_multi_class(outputs, masks, num_classes=3, ignore_background=False) * images.size(0)
                iou_total += iou_score_multi_class(outputs, masks, num_classes=3, ignore_background=False) * images.size(0)
                
                # Calculate per-class metrics for detailed analysis
                per_class_dice, per_class_iou = compute_per_class_metrics(outputs, masks, num_classes=3)
                for class_id in range(3):
                    per_class_dice_total[class_id] += per_class_dice[class_id] * images.size(0)
                    per_class_iou_total[class_id] += per_class_iou[class_id] * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice = dice_total / len(val_loader.dataset)
        val_iou = iou_total / len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Calculate average per-class metrics
        avg_per_class_dice = {k: v / len(val_loader.dataset) for k, v in per_class_dice_total.items()}
        avg_per_class_iou = {k: v / len(val_loader.dataset) for k, v in per_class_iou_total.items()}

        print(f"📊 Epoch {epoch+1}: "
            f"Train Loss = {epoch_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Dice (all classes) = {val_dice:.4f}, "
            f"Val IoU (all classes) = {val_iou:.4f}")
        print(f"   Per-class Dice: Background={avg_per_class_dice[0]:.4f}, "
              f"Muscle1={avg_per_class_dice[1]:.4f}, Muscle2={avg_per_class_dice[2]:.4f}")
        print(f"   Per-class IoU: Background={avg_per_class_iou[0]:.4f}, "
              f"Muscle1={avg_per_class_iou[1]:.4f}, Muscle2={avg_per_class_iou[2]:.4f}")


        # ✅ Save best model and early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset early stopping counter
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'unet_muscle_3_classes.pth'))
            print("✅ Saved best model")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")
        
        # ✅ Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'unet_muscle_3_classes_epoch_{epoch+1}.pth')
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
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve_3_classes.png'))
    plt.show()

if __name__ == '__main__':
    main()