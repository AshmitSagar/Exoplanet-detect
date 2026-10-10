"""
astronet_cnn.py — Dual-Branch 1D CNN for Exoplanet Transit Classification

Inspired by AstroNet (Shallue & Vanderburg, 2018) but independently designed
for modern PyTorch practices and future extensibility.

Architecture overview
─────────────────────
  global_view (B, 1, 2001)          local_view (B, 1, 201)
         │                                  │
  GlobalBranch                       LocalBranch
  5 × ConvBlock                      4 × ConvBlock
  AdaptiveAvgPool → (B, 256)         AdaptiveAvgPool → (B, 128)
         │                                  │
         └──────────── concat ──────────────┘
                          │
                    (B, 256 + 128)
                          │
                    Classifier
                    FC → BN → GELU → Dropout
                    FC → BN → GELU → Dropout
                    FC → logits (4)
                          │
                    [PC, AFP, NTP, UNK]

Labels
──────
  0 → PC   (Planet Candidate)
  1 → AFP  (Astrophysical False Positive)
  2 → NTP  (Non-Transiting Phenomenon)
  3 → UNK  (Unknown)

Notes
──────
  • Raw logits are returned — no Softmax. Use nn.CrossEntropyLoss.
  • AdaptiveAvgPool removes any dependence on input sequence length.
  • ConvBlock is a self-contained unit that can be swapped for residual
    or SE blocks without touching the branch or classifier code.
  • All intermediate sizes flow through config dicts, so the architecture
    is fully tunable from a single config object.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AstroNetConfig:
    """
    Single source of truth for every architectural hyper-parameter.
    Pass a modified config to explore different model sizes.

    Args:
        num_classes         : number of output classes (PC / AFP / NTP / UNK)
        global_channels     : conv channel progression for the global branch
        local_channels      : conv channel progression for the local branch
        global_pool_size    : AdaptiveAvgPool output length for global branch
        local_pool_size     : AdaptiveAvgPool output length for local branch
        classifier_hidden   : FC layer widths in the classifier head
        dropout_rate        : dropout probability applied in the classifier
        activation          : "gelu" (default) or "relu"
    """
    num_classes      : int        = 4
    global_channels  : List[int]  = field(default_factory=lambda: [16, 32, 64, 128, 256])
    local_channels   : List[int]  = field(default_factory=lambda: [16, 32, 64, 128])
    global_pool_size : int        = 1   # pool entire sequence → one vector
    local_pool_size  : int        = 1
    classifier_hidden: List[int]  = field(default_factory=lambda: [512, 256])
    dropout_rate     : float      = 0.5
    activation       : str        = "gelu"


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Reusable 1-D convolution block:
        Conv1d → BatchNorm1d → Activation → MaxPool1d

    Designed as a drop-in unit. To add residual connections or
    squeeze-and-excitation later, subclass or wrap this block —
    the branch stacks are built from a list of ConvBlocks so
    any replacement is local.

    Args:
        in_channels  : number of input channels
        out_channels : number of output channels
        kernel_size  : convolution kernel width (default 5)
        pool_size    : max-pool kernel/stride (default 2, set to 1 to skip)
        activation   : "gelu" or "relu"
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int = 5,
        pool_size   : int = 2,
        activation  : str = "gelu",
    ) -> None:
        super().__init__()

        act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

        layers: List[nn.Module] = [
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size = kernel_size,
                padding     = kernel_size // 2,   # 'same' padding
                bias        = False,              # BN absorbs the bias
            ),
            nn.BatchNorm1d(out_channels),
            act,
        ]
        if pool_size > 1:
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _build_branch(
    in_channels  : int,
    channel_list : List[int],
    pool_out_size: int,
    activation   : str,
) -> nn.Sequential:
    """
    Stack ConvBlocks for one branch, append AdaptiveAvgPool at the end.
    The adaptive pool decouples the branch from the input sequence length,
    which is required for ONNX export and future fine-tuning on different
    cadences (e.g., 30-min FFIs vs 2-min TPFs).
    """
    blocks: List[nn.Module] = []
    current_ch = in_channels

    for out_ch in channel_list:
        blocks.append(ConvBlock(current_ch, out_ch, activation=activation))
        current_ch = out_ch

    # Collapse the temporal dimension to `pool_out_size` steps
    # (1 → single feature vector per sample)
    blocks.append(nn.AdaptiveAvgPool1d(pool_out_size))

    return nn.Sequential(*blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class AstroNetCNN(nn.Module):
    """
    Dual-branch 1D CNN for exoplanet transit classification.

    Each branch independently extracts features from its view, the resulting
    feature vectors are concatenated, and a small FC classifier predicts
    one of [PC, AFP, NTP, UNK].

    Usage
    ─────
        cfg    = AstroNetConfig()
        model  = AstroNetCNN(cfg)
        logits = model(global_view, local_view)   # (B, 4)
        loss   = nn.CrossEntropyLoss()(logits, labels)

    Future extension hooks
    ───────────────────────
        • Attention / SE : replace ConvBlock in _build_branch channel_list
        • Residual       : subclass ConvBlock, override forward with skip
        • Transfer       : call model.freeze_branches() then fine-tune head
        • ONNX export    : torch.onnx.export(model, (gv, lv), "model.onnx")
    """

    def __init__(self, config: AstroNetConfig | None = None) -> None:
        super().__init__()
        cfg = config or AstroNetConfig()
        self.config = cfg

        act = cfg.activation

        # ── Branch 1: global view ────────────────────────────────────────────
        self.global_branch = _build_branch(
            in_channels   = 1,
            channel_list  = cfg.global_channels,
            pool_out_size = cfg.global_pool_size,
            activation    = act,
        )

        # ── Branch 2: local view ─────────────────────────────────────────────
        self.local_branch = _build_branch(
            in_channels   = 1,
            channel_list  = cfg.local_channels,
            pool_out_size = cfg.local_pool_size,
            activation    = act,
        )

        # ── Classifier head ──────────────────────────────────────────────────
        # Infer the concatenated feature size automatically so no manual
        # calculation is ever needed — works even if config changes.
        global_feat_size = cfg.global_channels[-1] * cfg.global_pool_size
        local_feat_size  = cfg.local_channels[-1]  * cfg.local_pool_size
        combined_size    = global_feat_size + local_feat_size

        self.classifier = self._build_classifier(
            in_features    = combined_size,
            hidden_widths  = cfg.classifier_hidden,
            num_classes    = cfg.num_classes,
            dropout_rate   = cfg.dropout_rate,
            activation     = act,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_classifier(
        in_features  : int,
        hidden_widths: List[int],
        num_classes  : int,
        dropout_rate : float,
        activation   : str,
    ) -> nn.Sequential:
        """
        Build the FC classifier as a Sequential so it is easy to inspect,
        replace, or freeze independently of the branches.
        """
        act_fn = nn.GELU if activation == "gelu" else nn.ReLU

        layers: List[nn.Module] = []
        current = in_features

        for width in hidden_widths:
            layers += [
                nn.Linear(current, width),
                nn.BatchNorm1d(width),
                act_fn(),
                nn.Dropout(p=dropout_rate),
            ]
            current = width

        layers.append(nn.Linear(current, num_classes))   # logit layer, no activation
        return nn.Sequential(*layers)

    # ── Transfer-learning utilities ───────────────────────────────────────────

    def freeze_branches(self) -> None:
        """Freeze both CNN branches. Fine-tune only the classifier head."""
        for param in self.global_branch.parameters():
            param.requires_grad = False
        for param in self.local_branch.parameters():
            param.requires_grad = False

    def unfreeze_branches(self) -> None:
        """Unfreeze branches for full fine-tuning."""
        for param in self.global_branch.parameters():
            param.requires_grad = True
        for param in self.local_branch.parameters():
            param.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        global_view: torch.Tensor,   # (B, 1, 2001)
        local_view : torch.Tensor,   # (B, 1, 201)
    ) -> torch.Tensor:               # (B, num_classes)  — raw logits
        """
        Forward pass.

        Args:
            global_view : phase-folded full-orbit light curve  (B, 1, 2001)
            local_view  : zoom on transit window               (B, 1, 201)

        Returns:
            logits      : un-normalised class scores           (B, num_classes)
                          Pass to nn.CrossEntropyLoss directly.
        """
        # Each branch: (B, 1, L) → (B, C, pool_size) → flatten → (B, C*pool)
        g = self.global_branch(global_view).flatten(start_dim=1)
        l = self.local_branch(local_view).flatten(start_dim=1)

        # Concatenate along feature dimension
        combined = torch.cat([g, l], dim=1)   # (B, global_feat + local_feat)

        return self.classifier(combined)       # (B, num_classes)

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Per-component parameter counts for architecture inspection."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            "global_branch" : _count(self.global_branch),
            "local_branch"  : _count(self.local_branch),
            "classifier"    : _count(self.classifier),
            "total"         : self.count_parameters(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Label utilities
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "PC", 1: "AFP", 2: "NTP", 3: "UNK"}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def logits_to_labels(logits: torch.Tensor) -> List[str]:
    """Convert raw logits to human-readable class names."""
    indices = logits.argmax(dim=-1).tolist()
    if isinstance(indices, int):
        indices = [indices]
    return [LABEL_MAP.get(i, "UNK") for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AstroNetCNN — self-test")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # ── Default config ────────────────────────────────────────────────────────
    cfg   = AstroNetConfig()
    model = AstroNetCNN(cfg).to(device)

    # ── Dummy batch ───────────────────────────────────────────────────────────
    B          = 8
    global_view = torch.randn(B, 1, 2001, device=device)
    local_view  = torch.randn(B, 1, 201,  device=device)

    print(f"\n  Input shapes")
    print(f"    global_view : {tuple(global_view.shape)}")
    print(f"    local_view  : {tuple(local_view.shape)}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(global_view, local_view)

    print(f"\n  Output")
    print(f"    logits shape : {tuple(logits.shape)}   (expected: ({B}, {cfg.num_classes}))")
    print(f"    predictions  : {logits_to_labels(logits)}")

    # ── Parameter breakdown ───────────────────────────────────────────────────
    breakdown = model.parameter_breakdown()
    print(f"\n  Parameters")
    for k, v in breakdown.items():
        print(f"    {k:<18}: {v:,}")

    # ── Loss check ────────────────────────────────────────────────────────────
    labels   = torch.randint(0, cfg.num_classes, (B,), device=device)
    loss_fn  = nn.CrossEntropyLoss()
    loss     = loss_fn(logits, labels)
    print(f"\n  Loss (random labels) : {loss.item():.4f}")

    # ── Gradient check ────────────────────────────────────────────────────────
    model.train()
    logits_train = model(global_view, local_view)
    loss_train   = loss_fn(logits_train, labels)
    loss_train.backward()
    print(f"  Backward pass        : OK")

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────
    model.freeze_branches()
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    model.unfreeze_branches()
    unfrozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"  freeze_branches()    : {frozen} params frozen")
    print(f"  unfreeze_branches()  : {unfrozen} params frozen")

    print("\n  All checks passed.")
    print("=" * 60)