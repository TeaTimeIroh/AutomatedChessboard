"""
Quantize chess_model.onnx → chess_model.espdl (INT8)
for deployment on ESP32-S3 with ESP-DL.

Run: python quantize_for_esp32.py
"""
import numpy as np

# ── CRITICAL: patch numpy BEFORE importing esp_ppq ───────────
# ESP-PPQ's espdl_exporter calls repr() on full weight tensors
# during export. With 6.75M parameters this hangs numpy's printer
# indefinitely. Setting threshold=128 makes numpy truncate arrays
# instead of printing every value. Must be done before esp_ppq import.
np.set_printoptions(threshold=128, edgeitems=4)

import torch
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx
import os

ONNX_PATH  = r"G:\ChessAI\LichessEvaluations\checkpoints\chess_model.onnx"
CALIB_PATH = r"G:\ChessAI\LichessEvaluations\calib_data.npy"
OUT_DIR    = r"G:\ChessAI\LichessEvaluations\esp32_model"
ESPDL_PATH = os.path.join(OUT_DIR, "chess_model.espdl")
TARGET     = "esp32s3"
DEVICE     = "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Calibration dataloader ────────────────────────────────────
print("Loading calibration data...")
calib_np = np.load(CALIB_PATH)              # (500, 18, 8, 8) float32
calib_t  = torch.from_numpy(calib_np)
dataset  = TensorDataset(calib_t)

def collate_fn(batch):
    return [torch.stack([b[0] for b in batch]).to(DEVICE)]

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
)

# ── Quantize ─────────────────────────────────────────────────
print(f"Quantizing: {ONNX_PATH}")
print(f"Target    : {TARGET}  |  INT8  |  {len(calib_np)} calibration samples")
print()

quant_graph = espdl_quantize_onnx(
    onnx_import_file  = ONNX_PATH,
    espdl_export_file = ESPDL_PATH,
    calib_dataloader  = dataloader,
    calib_steps       = 16,
    input_shape       = [1, 18, 8, 8],
    target            = TARGET,
    num_of_bits       = 8,
    collate_fn        = collate_fn,
    device            = DEVICE,
    error_report      = True,
    skip_export       = False,
    verbose           = 1,
)

# ── Report ───────────────────────────────────────────────────
print()
if os.path.exists(ESPDL_PATH):
    size_mb = os.path.getsize(ESPDL_PATH) / 1e6
    print("=" * 50)
    print("  QUANTIZATION COMPLETE")
    print("=" * 50)
    print(f"  chess_model.espdl : {size_mb:.2f} MB")
    print(f"  chess_model.json  : quantization config")
    print(f"  Saved to          : {OUT_DIR}")
else:
    print("ERROR: .espdl file was not created — check errors above")
