"""
muscle_binary_test.py

- Loads best or specified weights
- Computes BCE Loss, Dice, IoU for all test images
- Overall metrics and per-image results
- Optional save predictions as PNG
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import json

from unet import UNet
from muscle_dataset import MuscleMRIDataset
from efficiency_monitor import quick_check


def dice_score(pred, target, smooth: float = 1e-6):
    """pred/target: (B, 1, H, W)"""
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return dice


def iou_score(pred, target, smooth: float = 1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    iou = (inter + smooth) / (union + smooth)
    return iou


def main():
    # Hardcoded paths
    IMAGE_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\converted_dataset_binary\binary_dataset_class_90\test\images'
    MASK_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\converted_dataset_binary\binary_dataset_class_90\test\masks'
    SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\converted_dataset_binary\binary_dataset_class_90\splits\test.txt'
    CHECKPOINT = './checkpoints/unet_muscle_90_final.pth'
    RESULTS_DIR = './test_results_muscle_binary'
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    SAVE_PNG = True
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅  Using device: {device}")
    
    # Dataset
    target_size=(256, 256)
    test_dataset = MuscleMRIDataset(
        IMAGE_DIR, 
        MASK_DIR, 
        SPLIT_TXT, 
        transform=None,
        target_size=target_size,
        search_subfolders=True
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)
    
    # Model
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint '{CHECKPOINT}' not found!")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    print("✅ Loaded model state dict")
    
    model.eval()
    
    print("\n📊 Running efficiency check...")
    # Get the shape of the first sample from the dataset
    sample_image, _, _ = test_dataset[0]
    if isinstance(sample_image, torch.Tensor):
        h, w = sample_image.shape[-2:]
        c = sample_image.shape[0] if len(sample_image.shape) == 3 else 1
    else:
        h, w = sample_image.shape[:2]
        c = 1
    input_shape = (1, c, h, w)
    # Run efficiency check
    efficiency_results = quick_check(model, input_shape)
    print(f"\n✅ Model ready for testing")
    print(f"[INFO] Testing on {len(test_dataset)} images")
    
    # Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Create results directory
    results_dir = Path(RESULTS_DIR) / Path(CHECKPOINT).stem
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics storage
    csv_rows = [["filename", "dice", "iou", "bce_loss"]]
    
    # Overall metrics list
    dice_list, iou_list, loss_list = [], [], []
    
    # Prediction save directory
    if SAVE_PNG:
        pred_dir = results_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
    
    # Evaluation loop
    with torch.no_grad(), tqdm(total=len(test_loader), desc="🔍 Testing") as pbar:
        for batch_idx, batch_data in enumerate(test_loader):
            # Unpack data - now Dataset returns 3 values: image, mask, filename
            images, masks, filenames = batch_data
            
            images, masks = images.to(device), masks.to(device)
            
            # Forward propagation
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            # Calculate batch loss
            batch_loss = criterion(logits, masks)
            
            # Calculate metrics for each sample
            dice_scores = dice_score(probs, masks)
            iou_scores = iou_score(probs, masks)
            
            # Save predictions (if needed)
            if SAVE_PNG:
                preds_np = (probs.cpu().numpy() > 0.5).astype(np.uint8) * 255
                for i in range(preds_np.shape[0]):
                    filename = filenames[i]
                    
                    # Extract patient and slice numbers from filename
                    # Expected format: patient_001/slice_0000.png
                    name_without_ext = filename
                    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp', '.nii', '.nii.gz']:
                        if filename.endswith(ext):
                            name_without_ext = filename[:-len(ext)]
                            break
                    
                    # Parse path structure: patient_001\slice_0000 (Windows) or patient_001/slice_0000 (Unix)
                    # Handle both forward slashes and backslashes
                    path_parts = name_without_ext.replace('\\', '/').split('/')
                    patient_num = '000'
                    slice_num = '0000'
                    
                    if len(path_parts) >= 2:
                        # Extract patient number from patient_001
                        patient_part = path_parts[-2]  # patient_001
                        if patient_part.startswith('patient_'):
                            patient_num = patient_part[8:]  # Remove 'patient_' prefix
                        
                        # Extract slice number from slice_0000
                        slice_part = path_parts[-1]    # slice_0000
                        if slice_part.startswith('slice_'):
                            slice_num = slice_part[6:]  # Remove 'slice_' prefix
                    
                    # Create filename in format: 001_0000.png
                    fname = f"{patient_num}_{slice_num}.png"
                    cv2.imwrite(str(pred_dir / fname), preds_np[i, 0])
            
            # Process each sample
            for i in range(images.size(0)):
                filename = filenames[i]
                dice_val = dice_scores[i].item()
                iou_val = iou_scores[i].item()
                loss_val = batch_loss.item()  # Batch-level loss
                
                # Store overall metrics
                dice_list.append(dice_val)
                iou_list.append(iou_val)
                loss_list.append(loss_val)
                csv_rows.append([filename, f"{dice_val:.4f}", 
                               f"{iou_val:.4f}", f"{loss_val:.6f}"])
            
            pbar.set_postfix(loss=batch_loss.item())
            pbar.update()
    
    # Calculate overall mean
    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(iou_list)
    mean_loss = np.mean(loss_list)
    
    csv_rows.append(["AVERAGE", f"{mean_dice:.4f}", f"{mean_iou:.4f}", 
                    f"{mean_loss:.6f}"])
    
    # Save CSV file
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {Path(CHECKPOINT).stem}")
    print(f"{'='*60}")
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   BCE Loss : {mean_loss:.6f}")
    print(f"   Mean Dice: {mean_dice:.4f} ± {np.std(dice_list):.4f}")
    print(f"   Mean IoU : {mean_iou:.4f} ± {np.std(iou_list):.4f}")
    
    
    print(f"\n📊 EFFICIENCY METRICS:")
    print(f"   Parameters: {efficiency_results['total_params']/1e6:.2f}M")
    print(f"   Inference Time: {efficiency_results['mean_time_ms']:.2f} ms")
    print(f"   FPS: {efficiency_results['fps']:.2f}")
    if 'gpu_memory_mb' in efficiency_results:
        print(f"   GPU Memory: {efficiency_results['gpu_memory_mb']:.1f} MB")
    
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"   • Overall metrics: metrics.csv")
    
    print(f"{'='*60}")
    
    # Save summary JSON
    summary = {
        "checkpoint": CHECKPOINT,
        "dataset": "Muscle-Binary",
        "split": os.path.basename(SPLIT_TXT),
        "overall_metrics": {
            "dice": float(mean_dice),
            "dice_std": float(np.std(dice_list)),
            "iou": float(mean_iou),
            "iou_std": float(np.std(iou_list)),
            "bce_loss": float(mean_loss),
            "n_samples": len(dice_list)
        },
        "efficiency": {
            "parameters_M": efficiency_results['total_params']/1e6,
            "inference_time_ms": efficiency_results['mean_time_ms'],
            "fps": efficiency_results['fps'],
            "gpu_memory_mb": efficiency_results.get('gpu_memory_mb', None)
        },
        "config": {
            "batch_size": BATCH_SIZE,
            "input_shape": list(input_shape)
        }
    }
    
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()