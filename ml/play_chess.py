"""
Chess GUI — Play Against Your Trained Model
============================================
Uses pygame for the board UI and python-chess for move validation.
The AI uses your trained TinyChessNet to pick moves.

Install:
    pip install pygame python-chess torch

Run:
    python play_chess.py

Put this script in the same folder as chess_model_final.pth

Controls:
    - Click a piece to select it (yellow highlight)
    - Click a destination square to move
    - Green dots show legal moves
    - R : reset / return to color picker
    - F : flip board view (does not change who plays AI)
    - U : undo your last move
    - Q : quit
"""

from __future__ import annotations
import sys
import time
import os
import math
import threading

import pygame
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# MODEL PATH — relative to this script's directory
# Put chess_model_final.pth in the same folder as play_chess.py
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "chess_model_final.pth")

# ─────────────────────────────────────────────────────────────
# DISPLAY — true fullscreen, DPI-aware on Windows
# ─────────────────────────────────────────────────────────────

import platform

pygame.init()

# Fixed layout — 1024x768 window (board + panel)
SQ     = 120          # 120 * 8 = 960px board
MARGIN = 28
INFO_W = 340
W      = SQ * 8 + MARGIN * 2 + INFO_W   # 960 + 56 + 340 = 1356
H      = SQ * 8 + MARGIN * 2            # 960 + 56 = 1016
SW, SH = W, H   # used only for centering, not needed in fixed mode

FPS = 60

COL = {
    "light"       : (240, 217, 181),
    "dark"        : (181, 136,  99),
    "selected"    : (247, 215,  74),
    "legal_dot"   : (100, 200, 100),
    "last_move"   : (205, 210, 106),
    "check"       : (220,  50,  50),
    "bg"          : ( 30,  28,  26),
    "panel_bg"    : ( 22,  21,  18),
    "panel_line"  : ( 50,  48,  44),
    "text_main"   : (230, 220, 200),
    "text_dim"    : (140, 130, 110),
    "text_accent" : (200, 170,  80),
    "btn_bg"      : ( 55,  52,  46),
    "btn_hover"   : ( 80,  76,  66),
    "btn_border"  : (100,  90,  70),
    "white_piece" : (255, 255, 255),
    "black_piece" : ( 30,  28,  26),
}

PIECE_UNICODE = {
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.KING,   chess.WHITE): "♔",
    (chess.PAWN,   chess.BLACK): "♟",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.KING,   chess.BLACK): "♚",
}

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + r, inplace=True)

class TinyChessNet(nn.Module):
    def __init__(self, channels=64, num_blocks=6):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower   = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(channels, 24, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(24)
        self.policy_fc   = nn.Linear(24 * 8 * 8, 4096)
    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        return self.policy_fc(p.flatten(1))

# ─────────────────────────────────────────────────────────────
# BOARD ENCODING
# ─────────────────────────────────────────────────────────────

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4, (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,(chess.KING,   chess.BLACK): 11,
}

def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    flip   = (board.turn == chess.BLACK)
    for sq, piece in board.piece_map().items():
        row = sq // 8; col = sq % 8
        if flip:
            row = 7 - row
            color = not piece.color
        else:
            color = piece.color
        planes[PIECE_TO_PLANE[(piece.piece_type, color)], row, col] = 1.0
    planes[12, :, :] = 1.0
    our, their = board.turn, not board.turn
    planes[13, :, :] = float(board.has_kingside_castling_rights(our))
    planes[14, :, :] = float(board.has_queenside_castling_rights(our))
    planes[15, :, :] = float(board.has_kingside_castling_rights(their))
    planes[16, :, :] = float(board.has_queenside_castling_rights(their))
    if board.ep_square is not None:
        r = board.ep_square // 8; c = board.ep_square % 8
        if flip: r = 7 - r
        planes[17, r, c] = 1.0
    return planes

# ─────────────────────────────────────────────────────────────
# AI ENGINE
# ─────────────────────────────────────────────────────────────

class ChessAI:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  AI device: {self.device}")
        self.model = TinyChessNet(channels=64, num_blocks=6)
        ckpt  = torch.load(model_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"  Model loaded from: {model_path}")

    def _run(self, board: chess.Board):
        enc    = encode_board(board)
        tensor = torch.from_numpy(enc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)[0]
            probs  = torch.softmax(logits, dim=0).cpu().numpy()
        return probs

    def _decode_labels(self, sorted_labels, board, probs, n=None):
        flip  = (board.turn == chess.BLACK)
        legal = set(board.legal_moves)
        moves = []

        # The training data (Lichess) uses Chess960 castling notation where
        # the king moves to the ROOK square (e1h1, e1a1, e8h8, e8a8).
        # python-chess legal_moves uses standard notation where the king moves
        # to its destination square (e1g1, e1c1, e8g8, e8c8).
        # This map translates Chess960 castling labels → standard castling moves.
        CASTLING_FIX = {
            (4,  7): chess.Move(4,  6),   # white kingside:  e1h1 → e1g1
            (4,  0): chess.Move(4,  2),   # white queenside: e1a1 → e1c1
            (60, 63): chess.Move(60, 62), # black kingside:  e8h8 → e8g8
            (60, 56): chess.Move(60, 58), # black queenside: e8a8 → e8c8
        }

        for label in sorted_labels:
            if n and len(moves) >= n:
                break
            from_sq = int(label) // 64
            to_sq   = int(label) % 64
            if flip:
                from_sq = (7 - from_sq // 8) * 8 + (from_sq % 8)
                to_sq   = (7 - to_sq   // 8) * 8 + (to_sq   % 8)

            # Build candidate moves to try in order
            candidates = []

            # Castling fix — remap king-to-rook to king-to-destination
            castle = CASTLING_FIX.get((from_sq, to_sq))
            if castle:
                candidates.append(castle)

            candidates.append(chess.Move(from_sq, to_sq))
            candidates.append(chess.Move(from_sq, to_sq, promotion=chess.QUEEN))

            already = [mv for mv, _ in moves]
            for m in candidates:
                if m in legal and m not in already:
                    moves.append((m, float(probs[label])))
                    break

        return moves

    def get_move(self, board: chess.Board):
        probs  = self._run(board)
        sorted_labels = np.argsort(probs)[::-1]
        moves = self._decode_labels(sorted_labels, board, probs, n=1)
        if moves:
            return moves[0]
        legal = list(board.legal_moves)
        return (legal[0], 0.0) if legal else (None, 0.0)

    def get_top_moves(self, board: chess.Board, n: int = 5):
        probs  = self._run(board)
        sorted_labels = np.argsort(probs)[::-1]
        return self._decode_labels(sorted_labels, board, probs, n=n)

# ─────────────────────────────────────────────────────────────
# FONT HELPERS — scaled to resolution
# ─────────────────────────────────────────────────────────────

def fnt(size: int, bold=False):
    return pygame.font.SysFont("Segoe UI", max(8, int(size * SQ / 80)), bold=bold)

def fnt_piece():
    return pygame.font.SysFont("Segoe UI Symbol", max(24, int(SQ * 0.65)))

def fnt_mono(size: int):
    return pygame.font.SysFont("Consolas", max(8, int(size * SQ / 80)))

# ─────────────────────────────────────────────────────────────
# COLOR PICKER SCREEN
# ─────────────────────────────────────────────────────────────

def color_picker_screen(surf: pygame.Surface) -> chess.Color:
    """
    Show a fullscreen color picker.
    Returns chess.WHITE or chess.BLACK — the color the PLAYER chose.
    """
    clock = pygame.time.Clock()
    W_, H_ = surf.get_size()

    title_font  = pygame.font.SysFont("Segoe UI", max(28, H_ // 20), bold=True)
    sub_font    = pygame.font.SysFont("Segoe UI", max(16, H_ // 40))
    piece_font  = pygame.font.SysFont("Segoe UI Symbol", max(60, H_ // 8))
    btn_font    = pygame.font.SysFont("Segoe UI", max(18, H_ // 30), bold=True)

    btn_w = W_ // 4
    btn_h = H_ // 5
    gap   = W_ // 16
    total = btn_w * 2 + gap
    bx_w  = W_ // 2 - total // 2          # left edge of white button
    bx_b  = bx_w + btn_w + gap            # left edge of black button
    by    = H_ // 2 - btn_h // 2 + H_ // 16

    while True:
        mx, my = pygame.mouse.get_pos()

        hover_w = bx_w <= mx <= bx_w + btn_w and by <= my <= by + btn_h
        hover_b = bx_b <= mx <= bx_b + btn_w and by <= my <= by + btn_h

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if hover_w: return chess.WHITE
                if hover_b: return chess.BLACK

        # Background
        surf.fill(COL["bg"])

        # Subtle grid pattern
        for gx in range(0, W_, 40):
            pygame.draw.line(surf, (38, 36, 32), (gx, 0), (gx, H_), 1)
        for gy in range(0, H_, 40):
            pygame.draw.line(surf, (38, 36, 32), (0, gy), (W_, gy), 1)

        # Title
        t1 = title_font.render("♟  Chess AI", True, COL["text_accent"])
        surf.blit(t1, t1.get_rect(center=(W_ // 2, H_ // 5)))

        t2 = sub_font.render("Choose your side", True, COL["text_dim"])
        surf.blit(t2, t2.get_rect(center=(W_ // 2, H_ // 5 + t1.get_height() + 12)))

        # White button
        wc = COL["btn_hover"] if hover_w else COL["btn_bg"]
        pygame.draw.rect(surf, wc,          (bx_w, by, btn_w, btn_h), border_radius=12)
        pygame.draw.rect(surf, COL["btn_border"], (bx_w, by, btn_w, btn_h), 2, border_radius=12)
        wp = piece_font.render("♔", True, (255, 255, 255))
        surf.blit(wp, wp.get_rect(center=(bx_w + btn_w // 2, by + btn_h // 2 - 18)))
        wl = btn_font.render("White", True, COL["text_main"])
        surf.blit(wl, wl.get_rect(center=(bx_w + btn_w // 2, by + btn_h - 28)))

        # Black button
        bc = COL["btn_hover"] if hover_b else COL["btn_bg"]
        pygame.draw.rect(surf, bc,          (bx_b, by, btn_w, btn_h), border_radius=12)
        pygame.draw.rect(surf, COL["btn_border"], (bx_b, by, btn_w, btn_h), 2, border_radius=12)
        bp = piece_font.render("♚", True, (200, 190, 170))
        surf.blit(bp, bp.get_rect(center=(bx_b + btn_w // 2, by + btn_h // 2 - 18)))
        bl = btn_font.render("Black", True, COL["text_main"])
        surf.blit(bl, bl.get_rect(center=(bx_b + btn_w // 2, by + btn_h - 28)))

        # Hint
        hint = sub_font.render("Q to quit", True, COL["text_dim"])
        surf.blit(hint, hint.get_rect(center=(W_ // 2, H_ - 30)))

        pygame.display.flip()
        clock.tick(FPS)

# ─────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────

def sq_to_pixel(sq: int, flipped: bool):
    col = sq % 8
    row = sq // 8
    if not flipped:
        row = 7 - row
    else:
        col = 7 - col
    x = MARGIN + col * SQ + SQ // 2
    y = MARGIN + row * SQ + SQ // 2
    return x, y

def pixel_to_sq(px, py, flipped: bool):
    col = (px - MARGIN) // SQ
    row = (py - MARGIN) // SQ
    if col < 0 or col > 7 or row < 0 or row > 7:
        return None
    sq_row = (7 - row) if not flipped else row
    sq_col = col       if not flipped else (7 - col)
    return sq_row * 8 + sq_col

def draw_board(surf, board, selected_sq, legal_sqs, last_move, flipped):
    lf = fnt(13)
    files = "abcdefgh"
    ranks = "87654321" if not flipped else "12345678"
    fls   = files       if not flipped else files[::-1]

    for row in range(8):
        for col in range(8):
            sq    = (7 - row) * 8 + col if not flipped else row * 8 + (7 - col)
            x, y  = MARGIN + col * SQ, MARGIN + row * SQ
            light = (row + col) % 2 == 0

            if selected_sq == sq:
                color = COL["selected"]
            elif last_move and sq in (last_move.from_square, last_move.to_square):
                color = COL["last_move"]
            elif board.is_check() and board.king(board.turn) == sq:
                color = COL["check"]
            else:
                color = COL["light"] if light else COL["dark"]

            pygame.draw.rect(surf, color, (x, y, SQ, SQ))

            if sq in legal_sqs:
                cx, cy = x + SQ // 2, y + SQ // 2
                if board.piece_at(sq):
                    pygame.draw.circle(surf, COL["legal_dot"], (cx, cy), SQ // 2 - 4, 4)
                else:
                    pygame.draw.circle(surf, COL["legal_dot"], (cx, cy), SQ // 8 + 3)

    for i, r in enumerate(ranks):
        lbl = lf.render(r, True, COL["text_dim"])
        surf.blit(lbl, (MARGIN - lbl.get_width() - 4,
                        MARGIN + i * SQ + SQ // 2 - lbl.get_height() // 2))

    for i, f in enumerate(fls):
        lbl = lf.render(f, True, COL["text_dim"])
        surf.blit(lbl, (MARGIN + i * SQ + SQ // 2 - lbl.get_width() // 2,
                        MARGIN + 8 * SQ + 4))

def draw_pieces(surf, board, flipped, drag_sq=None, drag_pos=None):
    pf = fnt_piece()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or sq == drag_sq:
            continue
        cx, cy = sq_to_pixel(sq, flipped)
        symbol = PIECE_UNICODE[(piece.piece_type, piece.color)]
        color  = COL["white_piece"] if piece.color == chess.WHITE else COL["black_piece"]
        shadow = pf.render(symbol, True, (0, 0, 0))
        surf.blit(shadow, shadow.get_rect(center=(cx+2, cy+2)))
        surf.blit(pf.render(symbol, True, color),
                  pf.render(symbol, True, color).get_rect(center=(cx, cy)))

    if drag_sq is not None and drag_pos is not None:
        piece = board.piece_at(drag_sq)
        if piece:
            symbol = PIECE_UNICODE[(piece.piece_type, piece.color)]
            color  = COL["white_piece"] if piece.color == chess.WHITE else COL["black_piece"]
            shadow = pf.render(symbol, True, (0, 0, 0))
            surf.blit(shadow, shadow.get_rect(center=(drag_pos[0]+2, drag_pos[1]+2)))
            surf.blit(pf.render(symbol, True, color),
                      pf.render(symbol, True, color).get_rect(center=drag_pos))

def draw_panel(surf, board, player_color, game_state,
               move_history, top_moves, think_time):
    px = MARGIN * 2 + SQ * 8
    pw = W - px
    pygame.draw.rect(surf, COL["panel_bg"], (px, 0, pw, H))
    pygame.draw.line(surf, COL["panel_line"], (px, 0), (px, H), 1)

    x = px + 12
    y = 14
    lh_s = max(14, SQ // 5)   # line height small
    lh_m = max(18, SQ // 4)   # line height medium

    # Title
    t = fnt(18, bold=True).render("♟  Chess AI", True, COL["text_accent"])
    surf.blit(t, (x, y)); y += t.get_height() + 8
    pygame.draw.line(surf, COL["panel_line"], (px+8, y), (px+pw-8, y)); y += 8

    # Status
    if game_state == "playing":
        if board.turn == player_color:
            ts, tc = "Your turn", COL["text_accent"]
        else:
            ts, tc = "AI thinking...", (180, 180, 220)
    elif game_state == "checkmate":
        winner = "You win! ♟" if board.turn != player_color else "AI wins!"
        ts, tc = f"Checkmate — {winner}", COL["text_accent"]
    elif game_state == "stalemate":
        ts, tc = "Stalemate — Draw", COL["text_dim"]
    else:
        ts, tc = game_state, COL["text_dim"]

    surf.blit(fnt(14).render(ts, True, tc), (x, y)); y += lh_m
    surf.blit(fnt(12).render(
        f"Move {board.fullmove_number}  ·  "
        f"{'White' if board.turn==chess.WHITE else 'Black'} to move",
        True, COL["text_dim"]), (x, y)); y += lh_s
    if think_time > 0:
        surf.blit(fnt(12).render(f"AI thought: {think_time:.2f}s",
                                 True, COL["text_dim"]), (x, y))
    y += lh_m
    pygame.draw.line(surf, COL["panel_line"], (px+8, y), (px+pw-8, y)); y += 8

    # Top moves
    surf.blit(fnt(14).render("AI top moves:", True, COL["text_main"]), (x, y))
    y += lh_m
    if top_moves:
        bar_max = pw - 24
        for i, (move, conf) in enumerate(top_moves[:5]):
            bw = int(bar_max * min(conf * 4, 1.0))
            bc = (60, 120, 80) if i == 0 else (45, 65, 55)
            pygame.draw.rect(surf, bc, (x, y+2, bw, lh_s - 2), border_radius=3)
            mv = fnt_mono(13).render(
                f"{i+1}. {move.uci():<7} {conf*100:5.1f}%",
                True, COL["text_main"] if i == 0 else COL["text_dim"])
            surf.blit(mv, (x+4, y)); y += lh_m
    else:
        surf.blit(fnt(12).render("(make a move first)", True, COL["text_dim"]), (x, y))
        y += lh_m

    y += 6
    pygame.draw.line(surf, COL["panel_line"], (px+8, y), (px+pw-8, y)); y += 8

    # Move history
    surf.blit(fnt(14).render("Move history:", True, COL["text_main"]), (x, y))
    y += lh_m

    max_hist = min(len(move_history), 24)
    start    = max(0, len(move_history) - max_hist)
    visible  = move_history[start:]
    for i in range(0, len(visible), 2):
        num    = (start + i) // 2 + 1
        w_mv   = visible[i]     if i   < len(visible) else ""
        b_mv   = visible[i+1]   if i+1 < len(visible) else ""
        line   = fnt_mono(12).render(f"{num:>2}. {w_mv:<7} {b_mv}", True, COL["text_dim"])
        surf.blit(line, (x, y)); y += lh_s + 2
        if y > H - 70:
            break

    # Controls
    y = H - 58
    pygame.draw.line(surf, COL["panel_line"], (px+8, y), (px+pw-8, y)); y += 6
    for hint in ["R: new game   F: flip view", "U: undo move   Q: quit"]:
        surf.blit(fnt(12).render(hint, True, COL["text_dim"]), (x, y))
        y += lh_s + 2

# ─────────────────────────────────────────────────────────────
# PROMOTION DIALOG
# ─────────────────────────────────────────────────────────────

def ask_promotion(surf, color):
    clock  = pygame.font.SysFont("Segoe UI Symbol", max(40, SQ // 2))
    font2  = fnt(14)
    pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    syms   = ({chess.QUEEN:"♕", chess.ROOK:"♖", chess.BISHOP:"♗", chess.KNIGHT:"♘"}
              if color == chess.WHITE else
              {chess.QUEEN:"♛", chess.ROOK:"♜", chess.BISHOP:"♝", chess.KNIGHT:"♞"})

    bw, bh  = SQ + 10, SQ + 20
    total_w = bw * 4 + 12
    ox = (W - INFO_W) // 2 - total_w // 2
    oy = H // 2 - bh // 2

    pf     = pygame.font.SysFont("Segoe UI Symbol", max(40, SQ // 2))
    tick   = pygame.time.Clock()

    while True:
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, pt in enumerate(pieces):
                    bx = ox + i * (bw + 4)
                    if bx <= mx <= bx+bw and oy <= my <= oy+bh:
                        return pt

        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surf.blit(overlay, (0, 0))
        pygame.draw.rect(surf, COL["panel_bg"],
                         (ox-12, oy-36, total_w+24, bh+60), border_radius=10)
        title = font2.render("Promote to:", True, COL["text_main"])
        surf.blit(title, title.get_rect(center=(ox + total_w//2, oy - 20)))

        for i, pt in enumerate(pieces):
            bx    = ox + i * (bw+4)
            hover = bx <= mx <= bx+bw and oy <= my <= oy+bh
            pygame.draw.rect(surf, COL["btn_hover"] if hover else COL["btn_bg"],
                             (bx, oy, bw, bh), border_radius=8)
            sym = pf.render(syms[pt], True,
                            COL["white_piece"] if color==chess.WHITE else COL["black_piece"])
            surf.blit(sym, sym.get_rect(center=(bx+bw//2, oy+bh//2)))

        pygame.display.flip()
        tick.tick(FPS)

# ─────────────────────────────────────────────────────────────
# MAIN GAME
# ─────────────────────────────────────────────────────────────

def play(surf, ai, player_color: chess.Color):
    """
    Main game loop. player_color is the color the human plays.
    AI plays the opposite color and moves first if it has WHITE.
    """
    clock = pygame.time.Clock()

    # Board is shown from player's perspective by default
    flipped     = (player_color == chess.BLACK)
    board       = chess.Board()
    selected_sq = None
    legal_sqs   = set()
    last_move   = None
    game_state  = "playing"
    move_history= []
    top_moves   = []
    think_time  = 0.0
    ai_thinking = False
    drag_sq     = None
    drag_pos    = None

    def check_over():
        if board.is_checkmate():             return "checkmate"
        if board.is_stalemate():             return "stalemate"
        if board.is_insufficient_material(): return "draw (insufficient material)"
        if board.is_seventyfive_moves():     return "draw (75-move rule)"
        if board.is_fivefold_repetition():   return "draw (repetition)"
        return "playing"

    def do_ai_move():
        nonlocal last_move, game_state, ai_thinking, think_time, top_moves
        ai_thinking = True
        t0 = time.time()
        top_moves   = ai.get_top_moves(board, n=5)
        move, conf  = ai.get_move(board)
        think_time  = time.time() - t0
        if move and move in board.legal_moves:
            san = board.san(move)
            board.push(move)
            move_history.append(san)
            last_move  = move
            game_state = check_over()
        ai_thinking = False

    def trigger_ai():
        t = threading.Thread(target=do_ai_move, daemon=True)
        t.start()

    def apply_player_move(move):
        nonlocal last_move, game_state, selected_sq, legal_sqs
        san = board.san(move)
        board.push(move)
        move_history.append(san)
        last_move   = move
        selected_sq = None
        legal_sqs   = set()
        game_state  = check_over()
        if game_state == "playing":
            trigger_ai()

    # If AI is WHITE it moves first
    ai_color = not player_color
    if ai_color == chess.WHITE:
        trigger_ai()

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    return "menu"          # go back to color picker
                elif event.key == pygame.K_f:
                    flipped = not flipped
                    selected_sq = None; legal_sqs = set()
                elif event.key == pygame.K_u:
                    if len(board.move_stack) >= 2 and not ai_thinking:
                        board.pop(); board.pop()
                        if move_history: move_history.pop()
                        if move_history: move_history.pop()
                        last_move  = board.peek() if board.move_stack else None
                        game_state = "playing"
                        selected_sq = None; legal_sqs = set()
                        top_moves   = []

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game_state != "playing" or ai_thinking:
                    continue
                if board.turn != player_color:
                    continue
                mx, my = event.pos
                sq = pixel_to_sq(mx, my, flipped)
                if sq is None:
                    continue
                piece = board.piece_at(sq)
                if selected_sq is None:
                    if piece and piece.color == player_color:
                        selected_sq = sq
                        legal_sqs   = {m.to_square for m in board.legal_moves
                                       if m.from_square == sq}
                        drag_sq = sq; drag_pos = (mx, my)
                else:
                    if sq == selected_sq:
                        selected_sq = None; legal_sqs = set(); drag_sq = None
                    elif piece and piece.color == player_color:
                        selected_sq = sq
                        legal_sqs   = {m.to_square for m in board.legal_moves
                                       if m.from_square == sq}
                        drag_sq = sq; drag_pos = (mx, my)
                    elif sq in legal_sqs:
                        promo = None
                        mp = board.piece_at(selected_sq)
                        if mp and mp.piece_type == chess.PAWN:
                            if (player_color == chess.WHITE and sq//8 == 7) or \
                               (player_color == chess.BLACK and sq//8 == 0):
                                promo = ask_promotion(surf, player_color)
                        move = chess.Move(selected_sq, sq, promotion=promo)
                        if move in board.legal_moves:
                            apply_player_move(move)
                        drag_sq = None

            elif event.type == pygame.MOUSEBUTTONUP:
                if drag_sq is not None:
                    mx, my = event.pos
                    sq = pixel_to_sq(mx, my, flipped)
                    if sq and sq != drag_sq and sq in legal_sqs:
                        promo = None
                        mp = board.piece_at(drag_sq)
                        if mp and mp.piece_type == chess.PAWN:
                            if (player_color == chess.WHITE and sq//8 == 7) or \
                               (player_color == chess.BLACK and sq//8 == 0):
                                promo = ask_promotion(surf, player_color)
                        move = chess.Move(drag_sq, sq, promotion=promo)
                        if move in board.legal_moves:
                            selected_sq = drag_sq
                            legal_sqs   = {m.to_square for m in board.legal_moves
                                           if m.from_square == drag_sq}
                            apply_player_move(move)
                    drag_sq = None; drag_pos = None

            elif event.type == pygame.MOUSEMOTION:
                if drag_sq is not None:
                    drag_pos = event.pos

        # ── Draw ─────────────────────────────────────────────
        surf.fill(COL["bg"])
        draw_board(surf, board, selected_sq, legal_sqs, last_move, flipped)
        draw_pieces(surf, board, flipped, drag_sq, drag_pos)
        draw_panel(surf, board, player_color, game_state,
                   move_history, top_moves, think_time)

        # Thinking spinner
        if ai_thinking:
            t_sec  = pygame.time.get_ticks() / 1000
            cx, cy = MARGIN + 4 * SQ, MARGIN + 4 * SQ
            for i in range(8):
                a   = math.radians(t_sec * 360 + i * 45)
                r   = max(16, SQ // 4)
                px2 = int(cx + r * math.cos(a))
                py2 = int(cy + r * math.sin(a))
                alpha = int(255 * (i + 1) / 8)
                pygame.draw.circle(surf, (*COL["text_accent"][:3],), (px2, py2),
                                   max(3, SQ // 20))

        pygame.display.flip()

    return "quit"

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print(f"Place chess_model_final.pth in the same folder as play_chess.py")
        sys.exit(1)

    print("Loading AI model...")
    try:
        ai = ChessAI(MODEL_PATH)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    print("AI ready!")

    surf = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Chess AI")

    result = "menu"
    while result == "menu":
        player_color = color_picker_screen(surf)
        result       = play(surf, ai, player_color)

    pygame.quit()
    print("Thanks for playing!")

if __name__ == "__main__":
    main()
