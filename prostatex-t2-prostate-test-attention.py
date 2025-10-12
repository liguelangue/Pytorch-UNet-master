#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prostatex_t2_test_attention.py   (bug-fixed)
------------------------------------------------
• 自动读取 checkpoint 内的 model_config
• 若 checkpoint 里没有 config ⇒ 必须显式传入 --base_k_max / --bilinear 等参数
• 计算 BCE Loss / Dice / IoU，可选保存 PNG
"""

import os, argparse, cv2, numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet.unet_attention import AttentionUNet        # ★ 路径和训练脚本保持一致
from prostate_dataset   import ProstateMRIDataset    # ★

# -------------------- metric utils --------------------
def dice_score(pred, target, smooth=1e-6):
    pred   = (pred > 0.5).float()
    inter  = (pred * target).sum(dim=(1, 2, 3))
    union  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2*inter + smooth) / (union + smooth)).mean().item()

def iou_score(pred, target, smooth=1e-6):
    pred   = (pred > 0.5).float()
    inter  = (pred * target).sum(dim=(1, 2, 3))
    union  = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + smooth) / (union + smooth)).mean().item()

# -------------------- main --------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅  Using device: {device}")

    # -------- dataset --------
    test_ds = ProstateMRIDataset(args.image_dir, args.mask_dir, args.split_txt)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    # -------- checkpoint --------
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"❌  Checkpoint '{args.checkpoint}' 不存在")

    ckpt = torch.load(args.checkpoint, map_location=device)

    # -------- build model --------
    cfg = ckpt.get('model_config', {})                 # ★ 训练脚本里只有 best.pth 才包含
    # —— 如果 checkpoint 没有 config，就用 CLI 参数
    model = AttentionUNet(
        n_channels       = cfg.get('n_channels',  args.n_channels),
        n_classes        = cfg.get('n_classes',   args.n_classes),
        bilinear         = cfg.get('bilinear',    args.bilinear),
        base_k_max       = cfg.get('base_k_max',  args.base_k_max),
        use_attention    = cfg.get('use_attention', True),
        attention_layers = cfg.get('attention_layers', ['down3', 'down4'])
    ).to(device)

    # ★ 加载权重（严格匹配，报错更早、更清晰）
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # -------- loss --------
    criterion = nn.BCEWithLogitsLoss()

    # -------- eval loop --------
    test_loss, dice_sum, iou_sum = 0.0, 0.0, 0.0
    if args.save_png: os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad(), tqdm(total=len(test_loader), desc="🔍 Testing") as bar:
        for b, (imgs, masks, *extras) in enumerate(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)
            loss   = criterion(logits, masks)

            probs  = torch.sigmoid(logits)

            # accumulate
            bs            = imgs.size(0)
            test_loss    += loss.item()    * bs
            dice_sum     += dice_score(probs, masks) * bs
            iou_sum      += iou_score (probs, masks) * bs

            # optional save
            if args.save_png:
                bin_np = (probs.cpu().numpy() > 0.5).astype(np.uint8) * 255
                for i in range(bs):
                    if extras:                                   # ★ 若数据集返回路径
                        raw_name = os.path.basename(extras[0][i])
                        fname    = os.path.splitext(raw_name)[0] + "_pred.png"
                    else:
                        fname    = f"case{b:03d}_{i:02d}.png"
                    cv2.imwrite(os.path.join(args.save_dir, fname), bin_np[i, 0])

            bar.set_postfix(loss=f"{loss.item():.4f}")
            bar.update()

    n = len(test_loader.dataset)
    print("\n🎯  Test Summary")
    print(f"   BCE Loss : {test_loss / n:.6f}")
    print(f"   Mean Dice: {dice_sum  / n:.4f}")
    print(f"   Mean IoU : {iou_sum   / n:.4f}")

# -------------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("ProstateX-T2 AttentionUNet Test")

    # dataset
    p.add_argument("--image_dir",   required=True)
    p.add_argument("--mask_dir",    required=True)
    p.add_argument("--split_txt",   required=True)

    # checkpoint
    p.add_argument("--checkpoint",  required=True)

    # inference
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    # save
    p.add_argument("--save_png",  action="store_true")
    p.add_argument("--save_dir",  default="./pred_masks_attention")

    # manual overrides (仅在 checkpoint 无 config 时才生效)
    p.add_argument("--n_channels",  type=int, default=1)
    p.add_argument("--n_classes",   type=int, default=1)

    # ★ 默认值改成 **True**，跟训练脚本一致；如果想用反卷积，上 CLI 加 --no_bilinear
    bilinear_group = p.add_mutually_exclusive_group()
    bilinear_group.add_argument("--bilinear",    dest="bilinear", action="store_true",  default=True)
    bilinear_group.add_argument("--no_bilinear", dest="bilinear", action="store_false")

    p.add_argument("--base_k_max",  type=int, default=32)

    args = p.parse_args()
    main(args)
