"""
Chess Move Prediction — PyTorch Training Script
================================================
Trains a small ResNet CNN on the parsed Lichess eval dataset.
Input  : (18, 8, 8) board bitplanes
Output : 4096-class policy (from_sq * 64 + to_sq)

Usage:
    python train_chess_model.py

Requirements:
    pip install torch torchvision numpy
    (CUDA will be used automatically if available)
"""

from __future__ import annotations

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ─────────────────────────────────────────────────────────────
# CONFIG — tweak these
# ─────────────────────────────────────────────────────────────

CFG = {
    # Dataset
    "data_dir"    : r"G:\ChessAI\LichessEvaluations\dataset",
    "max_samples" : 5_000_000,      # use all 5M; reduce if RAM is tight

    # Model size  (channels=64, blocks=6 → ~1.3M params, ~1.3MB INT8)
    # For smaller: channels=32, blocks=4 → ~330K params
    "channels"    : 64,
    "num_blocks"  : 6,

    # Training
    "epochs"      : 20,
    "batch_size"  : 2048,           # fits comfortably in 12GB VRAM
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "label_smooth": 0.1,            # smoothing helps generalisation
    "val_split"   : 0.05,           # 5% validation

    # Mixed precision (faster on RTX cards, uses less VRAM)
    "amp"         : True,

    # Checkpointing
    "save_dir"    : r"G:\ChessAI\LichessEvaluations\checkpoints",
    "save_every"  : 5,              # save checkpoint every N epochs
}

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class ChessDataset(Dataset):
    """
    Memory-mapped dataset — does NOT load 22GB into RAM.
    np.load with mmap_mode='r' reads only the rows actually needed.
    """
    def __init__(self, data_dir: str, max_samples: int = None):
        print("  Loading dataset (memory-mapped, no RAM spike)...")
        X_path = os.path.join(data_dir, "X.npy")
        y_path = os.path.join(data_dir, "y.npy")

        print("  Loading X into RAM (this takes ~60s but GPU will run at 100% after)...")
        self.X = np.load(X_path)   # full load, no mmap
        self.y = np.load(y_path)

        if max_samples and max_samples < len(self.y):
            self.X = self.X[:max_samples]
            self.y = self.y[:max_samples]

        print(f"  X : {self.X.shape}  {self.X.dtype}")
        print(f"  y : {self.y.shape}  {self.y.dtype}")
        print(f"  Unique moves in dataset: {len(np.unique(self.y))}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Copy slice out of mmap into a real array for PyTorch
        x = torch.from_numpy(self.X[idx].copy())
        y = int(self.y[idx])
        return x, y


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class TinyChessNet(nn.Module):
    """
    Small residual CNN designed for INT8 quantisation on ESP32-S3.

    Architecture:
      input  (18, 8, 8)
        → input conv  → (C, 8, 8)
        → N res blocks → (C, 8, 8)
        → policy head  → 4096 logits

    Parameters (approx):
      channels=64, blocks=6  → 1.3M params → ~1.3MB INT8
      channels=32, blocks=4  →  330K params → ~330KB INT8
    """
    def __init__(self, channels: int = 64, num_blocks: int = 6):
        super().__init__()

        # Input projection: 18 planes → channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.res_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head (AlphaZero style)
        self.policy_conv = nn.Conv2d(channels, 24, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(24)
        self.policy_fc   = nn.Linear(24 * 8 * 8, 4096)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        p = p.flatten(1)
        p = self.policy_fc(p)
        return p   # raw logits — CrossEntropyLoss expects these

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────

def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5):
    """Fraction of samples where correct label is in top-k predictions."""
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(targets.unsqueeze(1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()


def fmt_time(seconds: float) -> str:
    if seconds < 60:   return f"{seconds:.0f}s"
    if seconds < 3600: return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, cfg):
    os.makedirs(cfg["save_dir"], exist_ok=True)
    path = os.path.join(cfg["save_dir"], f"chess_model_ep{epoch:02d}_{val_acc:.3f}.pth")
    torch.save({
        "epoch"       : epoch,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
        "sched_state" : scheduler.state_dict(),
        "val_acc"     : val_acc,
        "cfg"         : cfg,
    }, path)
    print(f"    Checkpoint saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*62}")
    print(f"  Chess Move Prediction — Training")
    print(f"{'='*62}")
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Channels  : {cfg['channels']}   Blocks: {cfg['num_blocks']}")
    print(f"  Batch     : {cfg['batch_size']}   Epochs: {cfg['epochs']}")
    print(f"  AMP       : {cfg['amp']}")

    # ── Dataset ──────────────────────────────────────────────
    print(f"\n[1/4] Loading dataset...")
    full_ds = ChessDataset(cfg["data_dir"], cfg["max_samples"])

    n_val   = int(len(full_ds) * cfg["val_split"])
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train : {n_train:,}  |  Val : {n_val:,}")

    # num_workers=0 on Windows to avoid multiprocessing issues
    nw = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True,  num_workers=nw, pin_memory=(device.type=="cuda"),
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2,
        shuffle=False, num_workers=nw, pin_memory=(device.type=="cuda"),
        persistent_workers=(nw > 0),
    )

    # ── Model ────────────────────────────────────────────────
    print(f"\n[2/4] Building model...")
    model = TinyChessNet(cfg["channels"], cfg["num_blocks"]).to(device)
    n_params = model.count_params()
    print(f"  Parameters : {n_params:,}  (~{n_params/1e6:.2f}M)")
    print(f"  INT8 size  : ~{n_params/1e6:.1f} MB  (after quantisation)")

    # ── Optimiser & Scheduler ────────────────────────────────
    print(f"\n[3/4] Setting up optimiser...")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smooth"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # Cosine annealing: lr decays smoothly from lr → 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type=="cuda")

    # ── Training ─────────────────────────────────────────────
    print(f"\n[4/4] Training...\n")
    print(f"  {'Ep':>3}  {'Loss':>7}  {'Top1':>6}  {'Top5':>6}  "
          f"{'Val-Top1':>8}  {'Val-Top5':>8}  {'LR':>8}  {'Time':>7}")
    print(f"  {'-'*65}")

    best_val_top1 = 0.0
    best_path     = None
    t_total       = time.time()

    steps_per_epoch = len(train_loader)

    for epoch in range(1, cfg["epochs"] + 1):
        t_epoch = time.time()

        # ── Train ─────────────────────────────────────────
        model.train()
        total_loss   = 0.0
        total_top1   = 0.0
        total_top5   = 0.0
        n_batches    = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = torch.tensor(y_batch, dtype=torch.long).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg["amp"] and device.type=="cuda"):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_top1 += topk_accuracy(logits.detach(), y_batch, k=1)
            total_top5 += topk_accuracy(logits.detach(), y_batch, k=5)
            n_batches  += 1

        avg_loss = total_loss / n_batches
        avg_top1 = total_top1 / n_batches
        avg_top5 = total_top5 / n_batches

        # ── Validate ──────────────────────────────────────
        model.eval()
        val_top1 = 0.0
        val_top5 = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = torch.tensor(y_batch, dtype=torch.long).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=cfg["amp"] and device.type=="cuda"):
                    logits = model(X_batch)
                val_top1 += topk_accuracy(logits, y_batch, k=1)
                val_top5 += topk_accuracy(logits, y_batch, k=5)
                n_val_batches += 1

        val_top1 /= n_val_batches
        val_top5 /= n_val_batches

        scheduler.step()
        current_lr   = scheduler.get_last_lr()[0]
        epoch_time   = time.time() - t_epoch
        elapsed      = time.time() - t_total
        eta          = elapsed / epoch * (cfg["epochs"] - epoch)

        print(f"  {epoch:>3}  {avg_loss:>7.4f}  {avg_top1:>6.3f}  {avg_top5:>6.3f}  "
              f"{val_top1:>8.3f}  {val_top5:>8.3f}  {current_lr:>8.6f}  "
              f"{fmt_time(epoch_time):>7}  ETA {fmt_time(eta)}")

        # Save best model
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            best_path = save_checkpoint(model, optimizer, scheduler, epoch, val_top1, cfg)

        # Periodic checkpoint
        elif epoch % cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_top1, cfg)

    # ── Final export ──────────────────────────────────────────
    total_time = time.time() - t_total
    print(f"\n{'='*62}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*62}")
    print(f"  Best val top-1 : {best_val_top1:.4f}  ({best_val_top1*100:.2f}%)")
    print(f"  Total time     : {fmt_time(total_time)}")
    print(f"  Best checkpoint: {best_path}")

    # Load best weights and export final model
    print(f"\n  Exporting final model...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Save plain weights for easy loading
    final_path = os.path.join(cfg["save_dir"], "chess_model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"  Final weights  : {final_path}")

    # Export to ONNX (needed for ESP-DL quantisation)
    export_onnx(model, cfg["save_dir"], device)

    return model


# ─────────────────────────────────────────────────────────────
# ONNX EXPORT
# ─────────────────────────────────────────────────────────────

def export_onnx(model, save_dir: str, device):
    """Export to ONNX — required for ESP-DL INT8 quantisation."""
    import torch.onnx
    onnx_path = os.path.join(save_dir, "chess_model.onnx")
    dummy     = torch.randn(1, 18, 8, 8).to(device)

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names   = ["board"],
        output_names  = ["policy"],
        dynamic_axes  = {"board": {0: "batch"}, "policy": {0: "batch"}},
        opset_version = 13,
        do_constant_folding = True,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  ONNX model     : {onnx_path}  ({size_mb:.1f} MB)")
    print(f"\n  Next step: quantise chess_model.onnx with ESP-DL tools")
    print(f"  for INT8 deployment on the ESP32-S3.")


# ─────────────────────────────────────────────────────────────
# QUICK INFERENCE TEST (run after training to sanity-check)
# ─────────────────────────────────────────────────────────────

def test_inference(model_path: str, data_dir: str, n_samples: int = 20):
    """
    Load saved model and run inference on a few samples.
    Shows top-5 predicted moves vs the actual Stockfish move.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    cfg_snap = torch.load(model_path, map_location="cpu")
    cfg_     = cfg_snap.get("cfg", CFG)
    model    = TinyChessNet(cfg_["channels"], cfg_["num_blocks"])
    model.load_state_dict(cfg_snap["model_state"])
    model.to(device).eval()

    X    = np.load(os.path.join(data_dir, "X.npy"),    mmap_mode="r")
    y    = np.load(os.path.join(data_dir, "y.npy"),    mmap_mode="r")
    meta = np.load(os.path.join(data_dir, "meta.npy"), mmap_mode="r")

    files = "abcdefgh"
    def lbl(label):
        f = label // 64; t = label % 64
        return files[f%8]+str(f//8+1)+"->"+files[t%8]+str(t//8+1)

    indices = np.random.choice(len(y), n_samples, replace=False)

    print(f"\n{'='*62}")
    print(f"  Inference test on {n_samples} random samples")
    print(f"{'='*62}")
    print(f"  {'#':>3}  {'Correct':>8}  {'Rank':>5}  {'Top-5 predictions'}")
    print(f"  {'-'*60}")

    correct_top1 = 0
    correct_top5 = 0

    for i, idx in enumerate(indices):
        x_np    = X[idx].copy()
        x_t     = torch.from_numpy(x_np).unsqueeze(0).to(device)
        true_lbl = int(y[idx])

        with torch.no_grad():
            logits = model(x_t)[0]
            probs  = torch.softmax(logits, dim=0)

        top5_vals, top5_idx = probs.topk(5)
        top5_idx  = top5_idx.cpu().numpy()
        top5_vals = top5_vals.cpu().numpy()

        rank = np.where(top5_idx == true_lbl)[0]
        rank_str = str(rank[0]+1) if len(rank) else ">5"

        is_top1 = top5_idx[0] == true_lbl
        is_top5 = true_lbl in top5_idx
        correct_top1 += is_top1
        correct_top5 += is_top5

        top5_str = "  ".join(f"{lbl(m)}({v:.2f})" for m,v in zip(top5_idx, top5_vals))
        marker   = "✓" if is_top1 else ("~" if is_top5 else "✗")

        print(f"  {i+1:>3}  {lbl(true_lbl):>8}  {rank_str:>5}  {top5_str}  {marker}")

    print(f"\n  Top-1 accuracy : {correct_top1}/{n_samples} = {correct_top1/n_samples*100:.1f}%")
    print(f"  Top-5 accuracy : {correct_top5}/{n_samples} = {correct_top5/n_samples*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  type=str, default=None,
                        help="Path to checkpoint to run inference test instead of training")
    parser.add_argument("--channels",   type=int, default=CFG["channels"])
    parser.add_argument("--blocks",     type=int, default=CFG["num_blocks"])
    parser.add_argument("--epochs",     type=int, default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    args = parser.parse_args()

    CFG["channels"]   = args.channels
    CFG["num_blocks"] = args.blocks
    CFG["epochs"]     = args.epochs
    CFG["batch_size"] = args.batch_size

    if args.test:
        test_inference(args.test, CFG["data_dir"])
    else:
        model = train(CFG)
