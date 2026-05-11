"""
Lichess Eval DB Parser & Feature Extractor
==========================================
Parses lichess_db_eval.jsonl.zst and produces:
  - X.npy : float32 array of shape (N, 18, 8, 8)  — board bitplanes
  - y.npy : int64  array of shape (N,)             — move labels (from*64 + to)
  - meta.npy : cp, depth, fullmove for each sample

Requirements:
    pip install python-chess zstandard numpy tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import chess
import numpy as np
import zstandard as zstd

# ─────────────────────────────────────────────
# Board encoding: 18 × 8 × 8 bitplanes
# ─────────────────────────────────────────────

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    flip = (board.turn == chess.BLACK)

    for square, piece in board.piece_map().items():
        row = square // 8
        col = square % 8
        if flip:
            row = 7 - row
            color = not piece.color
        else:
            color = piece.color
        plane = PIECE_TO_PLANE[(piece.piece_type, color)]
        planes[plane, row, col] = 1.0

    planes[12, :, :] = 1.0
    our_color   = board.turn
    their_color = not board.turn
    planes[13, :, :] = float(board.has_kingside_castling_rights(our_color))
    planes[14, :, :] = float(board.has_queenside_castling_rights(our_color))
    planes[15, :, :] = float(board.has_kingside_castling_rights(their_color))
    planes[16, :, :] = float(board.has_queenside_castling_rights(their_color))

    if board.ep_square is not None:
        row = board.ep_square // 8
        col = board.ep_square % 8
        if flip:
            row = 7 - row
        planes[17, row, col] = 1.0

    return planes


def move_to_label(move: chess.Move, flip: bool) -> int:
    from_sq = move.from_square
    to_sq   = move.to_square
    if flip:
        from_sq = (7 - from_sq // 8) * 8 + (from_sq % 8)
        to_sq   = (7 - to_sq   // 8) * 8 + (to_sq   % 8)
    return from_sq * 64 + to_sq


def label_to_move(label: int, flip: bool) -> chess.Move:
    from_sq = label // 64
    to_sq   = label % 64
    if flip:
        from_sq = (7 - from_sq // 8) * 8 + (from_sq % 8)
        to_sq   = (7 - to_sq   // 8) * 8 + (to_sq   % 8)
    return chess.Move(from_sq, to_sq)


# ─────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────

def should_keep(record: dict, board: chess.Board, best_eval: dict):
    depth = record.get("depth", 0)
    if depth < 20:
        return False, "low_depth"

    pvs = best_eval.get("pvs", [])
    if not pvs:
        return False, "no_pvs"

    top_pv = pvs[0]
    if "mate" in top_pv:
        if abs(top_pv["mate"]) <= 1:
            return False, "mate_in_1"
    elif "cp" in top_pv:
        if abs(top_pv["cp"]) > 1000:
            return False, "blowout_position"
    else:
        return False, "no_score"

    if sum(1 for _ in board.legal_moves) < 2:
        return False, "forced_move"

    return True, "ok"


# ─────────────────────────────────────────────
# Streaming parser
# ─────────────────────────────────────────────

def stream_jsonl_zst(filepath: str):
    """
    Generator yielding (record_dict, bytes_decompressed_so_far).
    Uses 16 MB read buffer for NVMe throughput.
    """
    READ_SIZE = 16 * 1024 * 1024  # 16 MB
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    bytes_read = 0

    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(READ_SIZE)
                if not chunk:
                    break
                bytes_read += len(chunk)
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line), bytes_read
                        except json.JSONDecodeError:
                            continue
            if buffer.strip():
                try:
                    yield json.loads(buffer), bytes_read
                except json.JSONDecodeError:
                    pass


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def print_progress(stats, max_samples, t0, bytes_read, bytes_total):
    elapsed   = time.time() - t0
    rate_rec  = stats["processed"] / max(elapsed, 0.001)
    rate_mb   = bytes_read / 1e6 / max(elapsed, 0.001)
    kept      = stats["kept"]
    keep_pct  = kept / max(stats["processed"], 1) * 100
    kept_rate = kept / max(elapsed, 0.001)
    eta_s     = (max_samples - kept) / max(kept_rate, 0.001)
    file_pct  = bytes_read / max(bytes_total, 1) * 100

    bar_width = 20
    filled    = int(bar_width * kept / max_samples)
    bar       = "#" * filled + "-" * (bar_width - filled)

    print(
        f"\r[{bar}] {kept:>8,}/{max_samples:,}  "
        f"| scanned {stats['processed']:>9,}  "
        f"| keep {keep_pct:4.1f}%  "
        f"| {rate_rec:>7,.0f} rec/s  "
        f"| {rate_mb:5.1f} MB/s  "
        f"| file {file_pct:5.1f}%  "
        f"| ETA {fmt_time(eta_s)}     ",
        end="", flush=True
    )


def print_filter_breakdown(stats):
    skip_keys = ["low_depth", "blowout_position", "mate_in_1",
                 "forced_move", "opening_start", "no_pvs",
                 "no_score", "bad_move", "other_skip"]
    total_skip = sum(stats.get(k, 0) for k in skip_keys)
    if total_skip == 0:
        return
    print(f"  Filter breakdown ({total_skip:,} skipped total):")
    for k in skip_keys:
        v = stats.get(k, 0)
        if v > 0:
            pct = v / total_skip * 100
            print(f"    {k:<22}: {v:>8,}  ({pct:4.1f}%)")


# ─────────────────────────────────────────────
# Main parse function
# ─────────────────────────────────────────────

def parse_dataset(
    input_path:  str,
    output_dir:  str,
    max_samples: int = 5_000_000,
    save_every:  int = 500_000,
    log_every:   int = 10_000,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bytes_total = os.path.getsize(input_path)

    stats = {
        "processed": 0, "kept": 0,
        "low_depth": 0, "blowout_position": 0, "mate_in_1": 0,
        "forced_move": 0, "opening_start": 0, "no_pvs": 0,
        "no_score": 0, "bad_move": 0, "other_skip": 0,
    }

    batch_X, batch_y, batch_meta = [], [], []
    chunk_idx = 0
    t0 = time.time()

    print("=" * 62)
    print("  Lichess Eval DB Parser")
    print("=" * 62)
    print(f"  Input    : {input_path}")
    print(f"  Size     : {bytes_total/1e9:.1f} GB (compressed)")
    print(f"  Output   : {output_dir}")
    print(f"  Target   : {max_samples:,} samples")
    print(f"  Filter   : depth>=20, |cp|<=1000, >=2 legal moves")
    print("=" * 62)
    print("  Opening decompression stream...")
    print("  (First records may take 10-30s on a large file)")
    print()
    sys.stdout.flush()

    first_seen = False

    for record, bytes_read in stream_jsonl_zst(input_path):
        if stats["kept"] >= max_samples:
            break

        stats["processed"] += 1

        # Confirm stream is alive on very first record
        if not first_seen:
            first_seen = True
            elapsed = time.time() - t0
            print(f"  Stream alive! First record after {elapsed:.1f}s")
            print(f"  Record keys : {list(record.keys())}")
            # Show a sample eval so we can verify the data looks right
            evals = record.get("evals", [])
            if evals:
                print(f"  Sample eval : depth={evals[0].get('depth','?')}  "
                      f"pvs={evals[0].get('pvs',[{}])[:1]}")
            print()
            sys.stdout.flush()

        try:
            fen   = record["fen"]
            evals = record.get("evals", [])
            if not evals:
                stats["no_pvs"] += 1
                continue

            best_eval        = max(evals, key=lambda e: e.get("depth", 0))
            record["depth"]  = best_eval.get("depth", 0)

            board = chess.Board(fen)
            keep, reason = should_keep(record, board, best_eval)
            if not keep:
                stats[reason] = stats.get(reason, 0) + 1
                continue

            pvs = best_eval["pvs"]

            def pv_score(pv):
                if "mate" in pv:
                    return 100000 - abs(pv["mate"]) if pv["mate"] > 0 else -100000
                return pv.get("cp", 0)

            pvs_sorted    = sorted(pvs, key=pv_score, reverse=True)
            best_pv_moves = pvs_sorted[0].get("line", "").split()
            if not best_pv_moves:
                stats["bad_move"] += 1
                continue

            move = chess.Move.from_uci(best_pv_moves[0])
            if move not in board.legal_moves:
                stats["bad_move"] += 1
                continue

            flip  = (board.turn == chess.BLACK)
            enc   = encode_board(board)
            label = move_to_label(move, flip)

            top_pv = pvs_sorted[0]
            if "mate" in top_pv:
                cp_val = 30000 if top_pv["mate"] > 0 else -30000
            else:
                cp_val = int(np.clip(top_pv.get("cp", 0), -32000, 32000))

            batch_X.append(enc)
            batch_y.append(label)
            batch_meta.append((cp_val, record["depth"], 0))
            stats["kept"] += 1

        except Exception as e:
            stats["other_skip"] += 1
            continue

        # Progress bar
        if stats["processed"] % log_every == 0:
            print_progress(stats, max_samples, t0, bytes_read, bytes_total)
            sys.stdout.flush()

        # Save chunk
        if len(batch_X) >= save_every:
            print()  # newline after progress bar
            _flush_chunk(batch_X, batch_y, batch_meta, output_dir, chunk_idx)
            chunk_idx += 1
            batch_X, batch_y, batch_meta = [], [], []
            print_filter_breakdown(stats)
            sys.stdout.flush()

    # Final flush
    if batch_X:
        print()
        _flush_chunk(batch_X, batch_y, batch_meta, output_dir, chunk_idx)
        chunk_idx += 1

    print(f"\n\nMerging {chunk_idx} chunk(s)...")
    sys.stdout.flush()
    _merge_chunks(output_dir, chunk_idx)

    elapsed = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  PARSE COMPLETE")
    print(f"{'=' * 62}")
    print(f"  Records scanned : {stats['processed']:>10,}")
    print(f"  Samples kept    : {stats['kept']:>10,}")
    print(f"  Keep rate       : {stats['kept']/max(stats['processed'],1)*100:.2f}%")
    print(f"  Time            : {fmt_time(elapsed)}")
    print(f"  Throughput      : {stats['processed']/elapsed:,.0f} rec/s")
    print()
    print_filter_breakdown(stats)
    print(f"\n  Output: {output_dir}/")
    print(f"    X.npy    (N, 18, 8, 8) float32")
    print(f"    y.npy    (N,)          int64")
    print(f"    meta.npy (N, 3)        int32  [cp, depth, fullmove]")


def _flush_chunk(batch_X, batch_y, batch_meta, output_dir, chunk_idx):
    X_arr    = np.array(batch_X,    dtype=np.float32)
    y_arr    = np.array(batch_y,    dtype=np.int64)
    meta_arr = np.array(batch_meta, dtype=np.int32)
    np.save(f"{output_dir}/X_chunk_{chunk_idx:04d}.npy",    X_arr)
    np.save(f"{output_dir}/y_chunk_{chunk_idx:04d}.npy",    y_arr)
    np.save(f"{output_dir}/meta_chunk_{chunk_idx:04d}.npy", meta_arr)
    print(f"  --> Chunk {chunk_idx:04d} saved: {len(batch_X):,} samples  "
          f"({X_arr.nbytes/1e6:.0f} MB)")


def _merge_chunks(output_dir, num_chunks):
    X_list, y_list, meta_list = [], [], []
    for i in range(num_chunks):
        X_list.append(np.load(f"{output_dir}/X_chunk_{i:04d}.npy"))
        y_list.append(np.load(f"{output_dir}/y_chunk_{i:04d}.npy"))
        meta_list.append(np.load(f"{output_dir}/meta_chunk_{i:04d}.npy"))
        print(f"  Loaded chunk {i+1}/{num_chunks}", end="\r", flush=True)

    print("\n  Concatenating ...", flush=True)
    X    = np.concatenate(X_list);    del X_list
    y    = np.concatenate(y_list);    del y_list
    meta = np.concatenate(meta_list); del meta_list

    print("  Shuffling ...", flush=True)
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    X, y, meta = X[perm], y[perm], meta[perm]

    print("  Saving ...", flush=True)
    np.save(f"{output_dir}/X.npy",    X)
    np.save(f"{output_dir}/y.npy",    y)
    np.save(f"{output_dir}/meta.npy", meta)

    for i in range(num_chunks):
        for prefix in ["X", "y", "meta"]:
            try:
                os.remove(f"{output_dir}/{prefix}_chunk_{i:04d}.npy")
            except FileNotFoundError:
                pass

    print(f"  X    : {X.shape}  {X.dtype}  {X.nbytes/1e9:.2f} GB")
    print(f"  y    : {y.shape}  {y.dtype}  {y.nbytes/1e6:.1f} MB")
    print(f"  meta : {meta.shape}  {meta.dtype}")


# ─────────────────────────────────────────────
# Dataset statistics helper
# ─────────────────────────────────────────────

def print_dataset_stats(output_dir: str):
    y    = np.load(f"{output_dir}/y.npy")
    meta = np.load(f"{output_dir}/meta.npy")
    cp_values = meta[:, 0]
    depths    = meta[:, 1]
    fullmoves = meta[:, 2]

    print(f"\n{'=' * 50}")
    print(f"  Dataset Statistics")
    print(f"{'=' * 50}")
    print(f"  Total samples : {len(y):,}")
    print(f"\n  CP score distribution:")
    for lo, hi in [(-1000,-500),(-500,-200),(-200,-50),(-50,50),
                   (50,200),(200,500),(500,1000)]:
        mask = (cp_values >= lo) & (cp_values < hi)
        print(f"    [{lo:>5} to {hi:>4}): {mask.sum():>8,}  ({mask.mean()*100:.1f}%)")
    print(f"\n  Depth distribution:")
    for d in [20, 25, 30, 35, 40]:
        mask = depths >= d
        print(f"    depth >= {d}: {mask.sum():>8,}  ({mask.mean()*100:.1f}%)")
    print(f"\n  Game phase:")
    print(f"    Opening  ( 1-10): {((fullmoves>=1)&(fullmoves<=10)).sum():>8,}")
    print(f"    Midgame  (11-30): {((fullmoves>=11)&(fullmoves<=30)).sum():>8,}")
    print(f"    Endgame  (31+  ): {(fullmoves>=31).sum():>8,}")
    print(f"\n  Unique moves seen: {len(np.unique(y))} / 4096")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Lichess eval DB")
    parser.add_argument("--input",       default="lichess_db_eval.jsonl.zst")
    parser.add_argument("--output_dir",  default="./dataset")
    parser.add_argument("--max_samples", type=int, default=5_000_000)
    parser.add_argument("--save_every",  type=int, default=500_000)
    parser.add_argument("--stats_only",  action="store_true")
    args = parser.parse_args()

    if args.stats_only:
        print_dataset_stats(args.output_dir)
    else:
        parse_dataset(
            input_path  = args.input,
            output_dir  = args.output_dir,
            max_samples = args.max_samples,
            save_every  = args.save_every,
        )
