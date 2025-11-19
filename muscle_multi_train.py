# muscle-multi-train.py

import os
import csv
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
from utils.eval import (
    hd95_numpy,
    compute_per_class_metrics,
    compute_per_class_hd95,
    compute_overall_hd95,
    dice_score_multi_class,
    iou_score_multi_class
)


def main():
    # ✅ Auto-select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # ✅ Data path settings
    IMAGE_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top10\train\images'
    MASK_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top10\train\masks'
    # TRAIN_SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top5\splits\train.txt'
    # VAL_SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top5\splits\val.txt'
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
        num_classes=11
    )
    val_dataset = MuscleMRIDatasetMulti(
        IMAGE_DIR, MASK_DIR,
        transform=None,
        target_size=target_size,
        search_subfolders=True,
        num_classes=11
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # ✅ Initialize model
    model = UNet(n_channels=1, n_classes=11, bilinear=True).to(device)
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
    val_dice_scores, val_iou_scores, val_hd95_scores = [], [], []
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

        # Validation phase - COMMENTED OUT
        # print("Validation started...")
        # model.eval()
        # val_loss = 0.0
        # dice_total = 0.0
        # iou_total = 0.0
        # hd95_total = 0.0
        # per_class_dice_total = {i: 0.0 for i in range(6)}
        # per_class_iou_total = {i: 0.0 for i in range(6)}
        # per_class_hd95_total = {i: 0.0 for i in range(6)}
        # hd95_count = 0  # Count valid HD95 values (excluding inf)
        # 
        # with torch.no_grad():
        #     for images, masks, filenames in val_loader:
        #         images, masks = images.to(device), masks.to(device)
        #         outputs = model(images)
        #         loss = criterion(outputs, masks)
        #         val_loss += loss.item() * images.size(0)
        # 
        #         # Calculate metrics including all classes (background + muscles)
        #         dice_total += dice_score_multi_class(outputs, masks, num_classes=6, ignore_background=False) * images.size(0)
        #         iou_total += iou_score_multi_class(outputs, masks, num_classes=6, ignore_background=False) * images.size(0)
        #         
        #         # Calculate per-class metrics for detailed analysis
        #         per_class_dice, per_class_iou = compute_per_class_metrics(outputs, masks, num_classes=6)
        #         
        #         # Calculate per-class HD95 (including background)
        #         per_class_hd95 = compute_per_class_hd95(outputs, masks, num_classes=6, ignore_index=None)
        #         
        #         # Calculate overall HD95 (treating all classes as one binary mask) - more memory efficient
        #         batch_hd95 = compute_overall_hd95(outputs, masks, ignore_background=False, num_classes=6)
        #         if not np.isinf(batch_hd95):
        #             hd95_total += batch_hd95 * images.size(0)
        #             hd95_count += images.size(0)
        #         
        #         for class_id in range(6):
        #             per_class_dice_total[class_id] += per_class_dice[class_id] * images.size(0)
        #             per_class_iou_total[class_id] += per_class_iou[class_id] * images.size(0)
        #             # HD95: accumulate only valid (non-inf) values
        #             if class_id in per_class_hd95 and not np.isinf(per_class_hd95[class_id]):
        #                 if class_id not in per_class_hd95_total:
        #                     per_class_hd95_total[class_id] = 0.0
        #                 per_class_hd95_total[class_id] += per_class_hd95[class_id] * images.size(0)
        # 
        # val_loss /= len(val_loader.dataset)
        # val_dice = dice_total / len(val_loader.dataset)
        # val_iou = iou_total / len(val_loader.dataset)
        # val_hd95 = hd95_total / hd95_count if hd95_count > 0 else float('inf')
        # val_losses.append(val_loss)
        # val_dice_scores.append(val_dice)
        # val_iou_scores.append(val_iou)
        # val_hd95_scores.append(val_hd95 if not np.isinf(val_hd95) else None)
        # scheduler.step(val_loss)
        # 
        # # Calculate average per-class metrics
        # avg_per_class_dice = {k: v / len(val_loader.dataset) for k, v in per_class_dice_total.items()}
        # avg_per_class_iou = {k: v / len(val_loader.dataset) for k, v in per_class_iou_total.items()}
        # avg_per_class_hd95 = {k: v / len(val_loader.dataset) if k in per_class_hd95_total and not np.isinf(v) else float('inf')
        #                       for k, v in per_class_hd95_total.items()}
        # 
        # print(f"📊 Epoch {epoch+1}: "
        #     f"Train Loss = {epoch_loss:.4f}, "
        #     f"Val Loss = {val_loss:.4f}, "
        #     f"Val Dice (all classes) = {val_dice:.4f}, "
        #     f"Val IoU (all classes) = {val_iou:.4f}, "
        #     f"Val HD95 (all classes) = {val_hd95:.4f}")
        # print(f"   Per-class Dice: Background={avg_per_class_dice[0]:.4f}, "
        #       f"Muscle1={avg_per_class_dice[1]:.4f}, Muscle2={avg_per_class_dice[2]:.4f}, "
        #       f"Muscle3={avg_per_class_dice[3]:.4f}, Muscle4={avg_per_class_dice[4]:.4f}, "
        #       f"Muscle5={avg_per_class_dice[5]:.4f}")
        # print(f"   Per-class IoU: Background={avg_per_class_iou[0]:.4f}, "
        #       f"Muscle1={avg_per_class_iou[1]:.4f}, Muscle2={avg_per_class_iou[2]:.4f}, "
        #       f"Muscle3={avg_per_class_iou[3]:.4f}, Muscle4={avg_per_class_iou[4]:.4f}, "
        #       f"Muscle5={avg_per_class_iou[5]:.4f}")
        # print(f"   Per-class HD95: Background={avg_per_class_hd95[0]:.4f}, "
        #       f"Muscle1={avg_per_class_hd95[1]:.4f}, Muscle2={avg_per_class_hd95[2]:.4f}, "
        #       f"Muscle3={avg_per_class_hd95[3]:.4f}, Muscle4={avg_per_class_hd95[4]:.4f}, "
        #       f"Muscle5={avg_per_class_hd95[5]:.4f}")

        # Simple print without validation metrics
        print(f"📊 Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}")
        
        # Use train loss for scheduler instead of val_loss
        scheduler.step(epoch_loss)

        # ✅ Save best model and early stopping check - COMMENTED OUT (requires validation)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0  # Reset early stopping counter
        #     torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'unet_muscle_5_classes.pth'))
        #     print("✅ Saved best model")
        # else:
        #     patience_counter += 1
        #     print(f"⏳ No improvement for {patience_counter} epochs")
        
        # Save model every epoch instead (since no validation)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'unet_muscle_10_classes.pth'))
        print("✅ Saved model")
        
        # ✅ Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'unet_muscle_10_classes_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
        
        # Early stopping check - COMMENTED OUT (requires validation)
        # if patience_counter >= patience:
        #     print(f"🛑 Early stopping triggered after {epoch+1} epochs")
        #     break

    # ✅ Visualize loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Val Loss')  # COMMENTED OUT - no validation
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve_10_classes.png'))
    plt.show()
    
    # ✅ Save training metrics to CSV
    metrics_file = os.path.join(CHECKPOINT_DIR, 'training_metrics_10_classes.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss'])  # Removed validation columns
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                f"{train_losses[epoch]:.6f}"
                # f"{val_losses[epoch]:.6f}",  # COMMENTED OUT
                # f"{val_dice_scores[epoch]:.4f}",  # COMMENTED OUT
                # f"{val_iou_scores[epoch]:.4f}",  # COMMENTED OUT
                # f"{val_hd95_scores[epoch]:.4f}" if val_hd95_scores[epoch] is not None else "inf"  # COMMENTED OUT
            ])
    print(f"💾 Training metrics saved to: {metrics_file}")

if __name__ == '__main__':
    main()