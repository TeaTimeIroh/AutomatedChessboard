"""
Convert chess_model.onnx → chess_model.tflite (full INT8)
Two-step approach:
  1. onnx2tf: ONNX → TF SavedModel  (no quantization yet)
  2. TFLiteConverter: SavedModel → INT8 TFLite  (proper calibration)

This separates conversion from quantization, which is more reliable.
The ONNX input name is "board" — this was the bug in the previous script.

Run in tflite_env: python convert_to_tflite.py
"""
import numpy as np
import os

ONNX_PATH    = r"G:\ChessAI\LichessEvaluations\checkpoints\chess_model.onnx"
CALIB_PATH   = r"G:\ChessAI\LichessEvaluations\calib_data.npy"
OUT_DIR      = r"G:\ChessAI\LichessEvaluations\esp32_model"
SAVED_MODEL  = os.path.join(OUT_DIR, "saved_model")
TFLITE_PATH  = os.path.join(OUT_DIR, "chess_model.tflite")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: ONNX → TF SavedModel (no quantization) ───────────
print("Step 1/3: ONNX → TF SavedModel (float32, no quantization)...")
print("  (takes ~2 minutes)")

import onnx2tf

onnx2tf.convert(
    input_onnx_file_path  = ONNX_PATH,
    output_folder_path    = SAVED_MODEL,
    non_verbose           = True,
)
print(f"  SavedModel written to: {SAVED_MODEL}")

# ── Step 2: Load calibration data ────────────────────────────
print("\nStep 2/3: Preparing calibration dataset...")
calib_np = np.load(CALIB_PATH).astype(np.float32)  # (500, 18, 8, 8) NCHW

# onnx2tf transposes the model to NHWC so we must transpose calib data too
calib_nhwc = calib_np.transpose(0, 2, 3, 1)        # → (500, 8, 8, 18) NHWC
print(f"  Calibration samples : {len(calib_nhwc)}")
print(f"  Input shape per sample: {calib_nhwc[0:1].shape}  (NHWC)")

def representative_dataset():
    """Yield one board position at a time as (1, 8, 8, 18) float32."""
    for i in range(len(calib_nhwc)):
        yield [calib_nhwc[i:i+1]]   # shape (1, 8, 8, 18)

# ── Step 3: SavedModel → INT8 TFLite ─────────────────────────
print("\nStep 3/3: SavedModel → INT8 TFLite (full quantization)...")
print("  (takes ~3 minutes)")
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)

# Full INT8 quantization — inputs AND outputs AND weights all INT8
converter.optimizations              = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset     = representative_dataset
converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type       = tf.int8
converter.inference_output_type      = tf.int8

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / 1e6
print(f"  Saved: {TFLITE_PATH}  ({size_mb:.2f} MB)")

# ── Verify ────────────────────────────────────────────────────
print("\nVerifying...")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"  Input  : shape={inp['shape']}  dtype={inp['dtype'].__name__}  scale={inp['quantization'][0]:.6f}")
print(f"  Output : shape={out['shape']}  dtype={out['dtype'].__name__}  scale={out['quantization'][0]:.6f}")

if inp['dtype'] == np.int8:
    print("  Full INT8 correct for ESP32-S3")
else:
    print("  Not INT8 — something went wrong")

# Test inference on empty board
sc_in,  zp_in  = inp['quantization']
sc_out, zp_out = out['quantization']
board_f32  = np.zeros((1, 8, 8, 18), dtype=np.float32)
board_int8 = (board_f32 / sc_in + zp_in).clip(-128,127).astype(np.int8)
interp.set_tensor(inp['index'], board_int8)
interp.invoke()
raw    = interp.get_tensor(out['index'])[0]
logits = (raw.astype(np.float32) - zp_out) * sc_out
top5   = np.argsort(logits)[::-1][:5]
files  = "abcdefgh"
sq     = lambda s: files[s%8]+str(s//8+1)
print(f"\n  Top-5 on empty board:")
for i, idx in enumerate(top5):
    print(f"    {i+1}. {sq(idx//64)}->{sq(idx%64)}  score={logits[idx]:.4f}")

print()
print("=" * 50)
print(f"  DONE — chess_model.tflite : {size_mb:.2f} MB  (full INT8)")
print("=" * 50)
