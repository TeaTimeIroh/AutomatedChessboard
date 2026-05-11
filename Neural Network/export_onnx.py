import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ── Rebuild model architecture (must match training) ──
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
        self.res_tower  = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(channels, 24, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(24)
        self.policy_fc   = nn.Linear(24 * 8 * 8, 4096)
    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        p = p.flatten(1)
        return self.policy_fc(p)

# ── Load checkpoint ──
ckpt_path = r"G:\ChessAI\LichessEvaluations\checkpoints\chess_model_ep20_0.373.pth"
save_dir  = r"G:\ChessAI\LichessEvaluations\checkpoints"

checkpoint = torch.load(ckpt_path, map_location="cpu")
model = TinyChessNet(channels=64, num_blocks=6)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("Checkpoint loaded successfully")

# ── Export to ONNX ──
dummy     = torch.randn(1, 18, 8, 8)
onnx_path = os.path.join(save_dir, "chess_model.onnx")

torch.onnx.export(
    model, dummy, onnx_path,
    input_names  = ["board"],
    output_names = ["policy"],
    dynamic_axes = {"board": {0: "batch"}, "policy": {0: "batch"}},
    opset_version = 13,
    do_constant_folding = True,
)

size_mb = os.path.getsize(onnx_path) / 1e6
print(f"ONNX exported: {onnx_path}  ({size_mb:.1f} MB)")

# ── Also save clean weights ──
weights_path = os.path.join(save_dir, "chess_model_final.pth")
torch.save(model.state_dict(), weights_path)
print(f"Clean weights: {weights_path}")