import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

# Try to import scipy for fastest HD95 calculation
try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import warnings
    warnings.warn("scipy not found. HD95 calculation will be slower. Install with: pip install scipy")


def hd95_numpy(pred: np.ndarray, gt: np.ndarray, percentile=95) -> float:
    """
    Calculate 95‑th percentile Hausdorff distance (pixel units).
    
    Time Complexity: O(n) where n = H * W (number of pixels).
    Uses scipy.ndimage.distance_transform_edt which implements an efficient
    O(n) algorithm for Euclidean distance transform, much faster than
    O(N_pred * N_gt) pairwise distance computation.
    
    pred / gt : 2‑D (H,W) array, bool or {0,1}
    
    Args:
        pred: Prediction binary mask
        gt: Ground truth binary mask
        percentile: Percentile to compute (default: 95)
    
    Returns:
        HD95 value in pixels
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # Special case handling
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    # Use distance transform method if scipy is available (much faster)
    if HAS_SCIPY:
        # Compute distance transforms
        # distance_transform_edt(~pred) computes Euclidean distance from each pixel 
        # to the nearest True pixel in pred (i.e., distance from background to foreground)
        pred_dt = distance_transform_edt(~pred)
        gt_dt = distance_transform_edt(~gt)
        
        # Get surface distances
        # For each True pixel in pred, get its distance to nearest True pixel in gt
        pred_surface_dists = gt_dt[pred]
        # For each True pixel in gt, get its distance to nearest True pixel in pred
        gt_surface_dists = pred_dt[gt]
        
        # Check if we have valid distances
        if len(pred_surface_dists) == 0 and len(gt_surface_dists) == 0:
            return 0.0
        
        # Combine and compute percentile
        all_surface_dists = np.concatenate([pred_surface_dists, gt_surface_dists])
        
        # If no distances (shouldn't happen if we passed the empty check above)
        if len(all_surface_dists) == 0:
            return 0.0
        
        # Compute percentile
        hd95_value = np.percentile(all_surface_dists, percentile)
        
        # Ensure we return a float (not a numpy scalar)
        return float(hd95_value)
    else:
        # Fallback to broadcast method if scipy is not available
        true_coords = np.argwhere(gt)      # (N_gt, 2)
        pred_coords = np.argwhere(pred)    # (N_pred, 2)

        # Nearest distance (pairwise Euclidean), broadcast implementation
        d_true_pred = np.min(
            np.linalg.norm(true_coords[:, None, :] - pred_coords[None, :, :], axis=-1),
            axis=1,
        )
        d_pred_true = np.min(
            np.linalg.norm(pred_coords[:, None, :] - true_coords[None, :, :], axis=-1),
            axis=1,
        )

        return np.percentile(np.hstack((d_true_pred, d_pred_true)), percentile)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


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


def compute_overall_hd95(pred, target, ignore_background=True, num_classes=3):
    """
    Compute overall HD95 treating all foreground classes as one binary mask.
    
    Time Complexity:
    - ignore_background=True: O(B * n) - single distance transform per batch item
    - ignore_background=False: O(C * B * n) - calls compute_per_class_hd95
    
    where B = batch size, C = num_classes, n = H * W pixels.
    Each distance_transform_edt call is O(n), so overall is linear in image size.
    
    Args:
        pred: [B, C, H, W] raw logits or [B, H, W] class indices
        target: [B, H, W] class indices
        ignore_background: If True, treat class 0 as background (exclude from foreground)
                          If False, compute average of per-class HD95 (including background)
        num_classes: Number of classes (used when ignore_background=False)
    
    Returns:
        float: Overall HD95 score (averaged across batch)
    """
    # Get hard predictions
    if pred.dim() == 4:  # [B, C, H, W]
        pred_soft = F.softmax(pred, dim=1)
        pred_hard = torch.argmax(pred_soft, dim=1)
    else:
        pred_hard = pred
    
    # Convert to numpy for HD95 calculation
    pred_np = pred_hard.cpu().numpy()  # [B, H, W]
    target_np = target.cpu().numpy()  # [B, H, W]
    
    if ignore_background:
        # Compute overall HD95 on foreground classes only
        hd95_values = []
        for b in range(pred_np.shape[0]):
            # Exclude background: only foreground classes (1, 2, ...)
            pred_foreground = (pred_np[b] > 0).astype(bool)
            target_foreground = (target_np[b] > 0).astype(bool)
            
            hd95_val = hd95_numpy(pred_foreground, target_foreground)
            # Skip infinite values when averaging
            if not np.isinf(hd95_val):
                hd95_values.append(hd95_val)
        
        if hd95_values:
            return np.mean(hd95_values)
        else:
            return 0.0
    else:
        # When ignore_background=False, compute average of per-class HD95 (including background)
        # This is more meaningful than trying to compute HD95 on all classes as one binary mask
        per_class_hd95 = compute_per_class_hd95(pred, target, num_classes=num_classes, ignore_index=None)
        
        # Average across all classes (including background)
        valid_hd95 = [v for v in per_class_hd95.values() if not np.isinf(v)]
        if valid_hd95:
            return np.mean(valid_hd95)
        else:
            return 0.0


def compute_per_class_hd95(pred, target, num_classes, ignore_index=None):
    """
    Compute per-class HD95 scores for detailed analysis.
    
    Time Complexity: O(C * B * n) where:
    - C = num_classes
    - B = batch size  
    - n = H * W (number of pixels)
    
    Each hd95_numpy call uses distance_transform_edt which is O(n),
    so total complexity is O(C * B * n) - still linear in image size.
    """
    per_class_hd95 = {}
    
    # Get hard predictions (single computation for all classes)
    if pred.dim() == 4:  # [B, C, H, W]
        pred_soft = F.softmax(pred, dim=1)
        pred_hard = torch.argmax(pred_soft, dim=1)
    else:
        pred_hard = pred
    
    # Convert to numpy once (avoid repeated conversions)
    pred_np = pred_hard.cpu().numpy()  # [B, H, W]
    target_np = target.cpu().numpy()  # [B, H, W]
    
    # Calculate HD95 for each class across all samples in batch
    # Note: We iterate over classes first, then batch items, to maintain
    # cache locality when processing the same class across batch
    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue
        
        hd95_values = []
        for b in range(pred_np.shape[0]):
            # Create binary masks for this class (O(n) - just comparison)
            pred_class = (pred_np[b] == class_id).astype(bool)
            target_class = (target_np[b] == class_id).astype(bool)
            
            # hd95_numpy uses distance_transform_edt: O(n) complexity
            hd95_val = hd95_numpy(pred_class, target_class)
            # Skip infinite values when averaging
            if not np.isinf(hd95_val):
                hd95_values.append(hd95_val)
        
        if hd95_values:
            per_class_hd95[class_id] = np.mean(hd95_values)
        else:
            # All samples had empty masks for this class
            per_class_hd95[class_id] = 0.0
    
    return per_class_hd95


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
