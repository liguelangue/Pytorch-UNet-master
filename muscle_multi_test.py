"""
muscle_multi_test.py

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import json

from unet import UNet
from muscle_dataset_multi import MuscleMRIDatasetMulti
from efficiency_monitor import quick_check
from utils.eval import (
    hd95_numpy,
    compute_per_class_metrics,
    compute_per_class_hd95,
    compute_overall_hd95,
    dice_score_multi_class,
    iou_score_multi_class
)


def main():
    # Hardcoded paths
    IMAGE_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top10\test\images'
    MASK_DIR = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top10\test\masks'
    SPLIT_TXT = r'C:\Users\DUDU\Documents\MIG\dataset\muscle_top10\splits\test.txt'
    CHECKPOINT = './checkpoints/unet_muscle_10_classes.pth'
    RESULTS_DIR = './test_results_muscle_10_classes'
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    SAVE_PNG = True
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅  Using device: {device}")
    
    # Dataset
    target_size=(256, 256)
    test_dataset = MuscleMRIDatasetMulti(
        IMAGE_DIR, 
        MASK_DIR, 
        transform=None,
        target_size=target_size,
        search_subfolders=True,
        num_classes=11  # 10 muscles + 1 background = 11 total classes
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)
    
    # Model
    model = UNet(n_channels=1, n_classes=11, bilinear=True).to(device)  # 11 classes: background + 10 muscles
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
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    
    # Create results directory
    results_dir = Path(RESULTS_DIR) / Path(CHECKPOINT).stem
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics storage
    csv_rows = [["filename", "dice_all_classes", "iou_all_classes", "hd95_all_classes", 
                 "hd95_class0", "hd95_class1", "hd95_class2", "hd95_class3", "hd95_class4", "hd95_class5",
                 "hd95_class6", "hd95_class7", "hd95_class8", "hd95_class9", "hd95_class10", 
                 "crossentropy_loss"]]
    
    # Overall metrics list
    dice_list, iou_list, hd95_list, loss_list = [], [], [], []
    per_class_hd95_all = {i: [] for i in range(11)}  # Track per-class HD95 across all images (11 classes)
    
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
            probs = torch.softmax(logits, dim=1)  # Use softmax for multi-class
            
            # Calculate batch loss
            batch_loss = criterion(logits, masks)
            
            # Save predictions (if needed)
            if SAVE_PNG:
                preds_np = torch.argmax(probs, dim=1).cpu().numpy()  # Get class predictions
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
                    
                    # Scale class predictions to original values (0, 21, 34, 76, 90, 96, 200, 214, 241, 248, 255)
                    pred_mask = preds_np[i].astype(np.uint8)
                    pred_mask[pred_mask == 1] = 21   # Muscle class 1
                    pred_mask[pred_mask == 2] = 34   # Muscle class 2
                    pred_mask[pred_mask == 3] = 76   # Muscle class 3
                    pred_mask[pred_mask == 4] = 90   # Muscle class 4
                    pred_mask[pred_mask == 5] = 96   # Muscle class 5
                    pred_mask[pred_mask == 6] = 200  # Muscle class 6
                    pred_mask[pred_mask == 7] = 214  # Muscle class 7
                    pred_mask[pred_mask == 8] = 241  # Muscle class 8
                    pred_mask[pred_mask == 9] = 248  # Muscle class 9
                    pred_mask[pred_mask == 10] = 255  # Muscle class 10
                    # Class 0 remains 0 (background)
                    
                    # Create filename in format: 001_0000.png
                    fname = f"{patient_num}_{slice_num}.png"
                    cv2.imwrite(str(pred_dir / fname), pred_mask)
            
            # Process each sample individually
            for i in range(images.size(0)):
                filename = filenames[i]
                
                # Extract single image and mask for per-image metric calculation
                single_logit = logits[i:i+1]  # Keep batch dimension [1, C, H, W]
                single_mask = masks[i:i+1]    # Keep batch dimension [1, H, W]
                
                # Calculate metrics for this single image (including all classes)
                dice_val = dice_score_multi_class(single_logit, single_mask, num_classes=11, ignore_background=False)
                iou_val = iou_score_multi_class(single_logit, single_mask, num_classes=11, ignore_background=False)
                loss_val = batch_loss.item()  # Batch-level loss
                
                # Calculate per-class HD95 (including background)
                per_class_hd95 = compute_per_class_hd95(single_logit, single_mask, num_classes=11, ignore_index=None)
                
                # Calculate overall HD95 (treating all classes as one binary mask) - more memory efficient
                hd95_val = compute_overall_hd95(single_logit, single_mask, ignore_background=False, num_classes=11)
                if np.isinf(hd95_val):
                    hd95_val = float('inf')
                
                # Store per-class HD95 values
                hd95_classes = [per_class_hd95.get(i, float('inf')) for i in range(11)]
                
                # Track per-class HD95 for overall statistics
                for class_id in range(11):
                    hd95_val_class = hd95_classes[class_id]
                    if not np.isinf(hd95_val_class):
                        per_class_hd95_all[class_id].append(hd95_val_class)
                
                # Store overall metrics
                dice_list.append(dice_val)
                iou_list.append(iou_val)
                hd95_list.append(hd95_val)
                loss_list.append(loss_val)
                csv_rows.append([filename, f"{dice_val:.4f}", 
                               f"{iou_val:.4f}", f"{hd95_val:.4f}" if not np.isinf(hd95_val) else "inf",
                               f"{hd95_classes[0]:.4f}" if not np.isinf(hd95_classes[0]) else "inf",
                               f"{hd95_classes[1]:.4f}" if not np.isinf(hd95_classes[1]) else "inf",
                               f"{hd95_classes[2]:.4f}" if not np.isinf(hd95_classes[2]) else "inf",
                               f"{hd95_classes[3]:.4f}" if not np.isinf(hd95_classes[3]) else "inf",
                               f"{hd95_classes[4]:.4f}" if not np.isinf(hd95_classes[4]) else "inf",
                               f"{hd95_classes[5]:.4f}" if not np.isinf(hd95_classes[5]) else "inf",
                               f"{hd95_classes[6]:.4f}" if not np.isinf(hd95_classes[6]) else "inf",
                               f"{hd95_classes[7]:.4f}" if not np.isinf(hd95_classes[7]) else "inf",
                               f"{hd95_classes[8]:.4f}" if not np.isinf(hd95_classes[8]) else "inf",
                               f"{hd95_classes[9]:.4f}" if not np.isinf(hd95_classes[9]) else "inf",
                               f"{hd95_classes[10]:.4f}" if not np.isinf(hd95_classes[10]) else "inf",
                               f"{loss_val:.6f}"])
            
            pbar.set_postfix(loss=batch_loss.item())
            pbar.update()
    
    # Calculate overall mean
    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(iou_list)
    # HD95: exclude infinite values when calculating mean
    hd95_valid = [h for h in hd95_list if not np.isinf(h)]
    mean_hd95 = np.mean(hd95_valid) if hd95_valid else float('inf')
    mean_loss = np.mean(loss_list)
    
    # Calculate per-class HD95 means
    mean_hd95_per_class = []
    for class_id in range(11):
        if per_class_hd95_all[class_id]:
            mean_hd95_per_class.append(np.mean(per_class_hd95_all[class_id]))
        else:
            mean_hd95_per_class.append(float('inf'))
    
    csv_rows.append(["AVERAGE", f"{mean_dice:.4f}", f"{mean_iou:.4f}", 
                    f"{mean_hd95:.4f}" if not np.isinf(mean_hd95) else "inf",
                    f"{mean_hd95_per_class[0]:.4f}" if not np.isinf(mean_hd95_per_class[0]) else "inf",
                    f"{mean_hd95_per_class[1]:.4f}" if not np.isinf(mean_hd95_per_class[1]) else "inf",
                    f"{mean_hd95_per_class[2]:.4f}" if not np.isinf(mean_hd95_per_class[2]) else "inf",
                    f"{mean_hd95_per_class[3]:.4f}" if not np.isinf(mean_hd95_per_class[3]) else "inf",
                    f"{mean_hd95_per_class[4]:.4f}" if not np.isinf(mean_hd95_per_class[4]) else "inf",
                    f"{mean_hd95_per_class[5]:.4f}" if not np.isinf(mean_hd95_per_class[5]) else "inf",
                    f"{mean_hd95_per_class[6]:.4f}" if not np.isinf(mean_hd95_per_class[6]) else "inf",
                    f"{mean_hd95_per_class[7]:.4f}" if not np.isinf(mean_hd95_per_class[7]) else "inf",
                    f"{mean_hd95_per_class[8]:.4f}" if not np.isinf(mean_hd95_per_class[8]) else "inf",
                    f"{mean_hd95_per_class[9]:.4f}" if not np.isinf(mean_hd95_per_class[9]) else "inf",
                    f"{mean_hd95_per_class[10]:.4f}" if not np.isinf(mean_hd95_per_class[10]) else "inf",
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
    print(f"   CrossEntropy Loss : {mean_loss:.6f}")
    print(f"   Mean Dice (all classes): {mean_dice:.4f} ± {np.std(dice_list):.4f}")
    print(f"   Mean IoU (all classes) : {mean_iou:.4f} ± {np.std(iou_list):.4f}")
    if not np.isinf(mean_hd95):
        print(f"   Mean HD95 (all classes)    : {mean_hd95:.4f} ± {np.std(hd95_valid):.4f}")
    else:
        print(f"   Mean HD95 (all classes)    : inf (no valid values)")
    # print(f"\n📊 PER-CLASS HD95:")
    # class_names = ["Background", "Muscle1", "Muscle2", "Muscle3", "Muscle4", "Muscle5"]
    # for class_id in range(6):
    #     if per_class_hd95_all[class_id]:
    #         mean_val = mean_hd95_per_class[class_id]
    #         std_val = np.std(per_class_hd95_all[class_id])
    #         print(f"   HD95 {class_names[class_id]} (class {class_id}): {mean_val:.4f} ± {std_val:.4f}")
    #     else:
    #         print(f"   HD95 {class_names[class_id]} (class {class_id}): inf (no valid values)")
    print(f"   Note: Metrics calculated for all classes (including background). HD95 computed per class.")
    
    
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
        "dataset": "Muscle-Multi-Class",
        "split": os.path.basename(SPLIT_TXT),
        "overall_metrics": {
            "dice_all_classes": float(mean_dice),
            "dice_all_classes_std": float(np.std(dice_list)),
            "iou_all_classes": float(mean_iou),
            "iou_all_classes_std": float(np.std(iou_list)),
            "hd95_all_classes": float(mean_hd95) if not np.isinf(mean_hd95) else None,
            "hd95_all_classes_std": float(np.std(hd95_valid)) if hd95_valid else None,
            "hd95_per_class": {
                f"class_{i}_{'background' if i == 0 else f'muscle{i}'}": {
                    "mean": float(mean_hd95_per_class[i]) if not np.isinf(mean_hd95_per_class[i]) else None,
                    "std": float(np.std(per_class_hd95_all[i])) if per_class_hd95_all[i] else None
                } for i in range(11)
            },
            "crossentropy_loss": float(mean_loss),
            "n_samples": len(dice_list),
            "note": "Metrics calculated for all classes (background class 0 + muscle classes 1-10). HD95 calculated per class including background."
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