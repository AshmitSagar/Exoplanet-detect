"""
config.py — TrainingConfig: single source of truth for every hyperparameter.

All fields have sensible defaults that work out-of-the-box with the 100-sample
Kepler NPZ dataset.  Override any field when constructing the config, or let
train.py parse them from the CLI.

Design notes
────────────
• Using a dataclass (not argparse Namespace) keeps the config serialisable,
  type-safe, and easy to pass between modules.
• All path fields are stored as pathlib.Path objects so downstream code never
  has to call str() or do manual path joining.
• AMP and num_workers default to sensible values that auto-adapt to the
  hardware detected at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_num_workers() -> int:
    """Use up to 4 workers, but never more than the physical CPU count."""
    return min(4, os.cpu_count() or 1)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Complete configuration for one training run.

    Data
    ────
    data_dir        : directory containing the converted .npz samples
                      (output of datasets/convert_to_npz.py)
    output_dir      : root output directory; sub-directories are created
                      automatically for checkpoints, logs, and plots.
    val_fraction    : fraction of the full dataset used for validation
    seed            : global random seed (Python, NumPy, PyTorch)

    DataLoader
    ──────────
    batch_size      : samples per mini-batch
    num_workers     : DataLoader worker processes (0 = load in main process)
    pin_memory      : copy tensors to CUDA pinned memory for faster transfers
                      (automatically set to True when CUDA is detected)

    Optimiser
    ─────────
    learning_rate   : initial AdamW learning rate
    weight_decay    : AdamW L2 regularisation coefficient

    Scheduler
    ─────────
    lr_scheduler    : "cosine" (CosineAnnealingLR) or "step" (StepLR)
    lr_step_size    : epochs between LR decay steps (StepLR only)
    lr_gamma        : multiplicative decay factor (StepLR only)
    lr_eta_min      : minimum LR for cosine schedule

    Training loop
    ─────────────
    num_epochs      : maximum number of training epochs
    mixed_precision : enable FP16 AMP — set to False if CUDA is absent
                      (train.py auto-overrides this based on torch.cuda.is_available())
    use_class_weights : reweight CrossEntropyLoss by inverse class frequency
                        in the training split (recommended for imbalanced data)

    Early stopping
    ──────────────
    patience        : epochs without val-loss improvement before stopping
    min_delta       : minimum improvement in val loss to count as progress

    Checkpointing
    ─────────────
    save_every      : save a "last_model.pth" checkpoint every N epochs
                      (0 = only save best and final)
    resume_from     : path to a checkpoint .pth file to resume from;
                      None means start from scratch

    Model
    ─────
    num_classes     : must match LABEL_MAP in kepler_dataset.py (4)
    dropout_rate    : passed through to AstroNetConfig
    """

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir     : Path = field(default_factory=lambda: Path("data/ai_ready/npz"))
    output_dir   : Path = field(default_factory=lambda: Path("data/models/training_run"))
    val_fraction : float = 0.20
    seed         : int   = 42

    # ── DataLoader ────────────────────────────────────────────────────────────
    batch_size   : int  = 32
    num_workers  : int  = field(default_factory=_default_num_workers)
    pin_memory   : bool = False   # overridden to True when CUDA is available

    # ── Optimiser ─────────────────────────────────────────────────────────────
    learning_rate : float = 1e-3
    weight_decay  : float = 1e-4

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    lr_scheduler  : str   = "cosine"   # "cosine" | "step"
    lr_step_size  : int   = 10         # epochs between steps (StepLR)
    lr_gamma      : float = 0.5        # decay factor (StepLR)
    lr_eta_min    : float = 1e-6       # cosine floor

    # ── Training loop ─────────────────────────────────────────────────────────
    num_epochs        : int  = 100
    mixed_precision   : bool = False   # overridden in train.py based on CUDA
    use_class_weights : bool = True

    # ── Early stopping ────────────────────────────────────────────────────────
    patience  : int   = 15
    min_delta : float = 1e-4

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_every  : int              = 5
    resume_from : Path | None = None

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes  : int   = 4
    dropout_rate : float = 0.5

    # ── Internal: derived dirs (populated by train.py after __post_init__) ────
    checkpoint_dir : Path = field(init=False)
    log_dir        : Path = field(init=False)
    plot_dir       : Path = field(init=False)

    def __post_init__(self) -> None:
        # Coerce str → Path in case the user passes strings
        self.data_dir    = Path(self.data_dir)
        self.output_dir  = Path(self.output_dir)
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)

        # Derived subdirectories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir        = self.output_dir / "logs"
        self.plot_dir       = self.output_dir / "plots"

    def create_output_dirs(self) -> None:
        """Create all output directories, no-op if they already exist."""
        for d in (self.checkpoint_dir, self.log_dir, self.plot_dir):
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        lines = ["TrainingConfig("]
        for f_name, f_val in self.__dict__.items():
            lines.append(f"  {f_name:<22} = {f_val!r}")
        lines.append(")")
        return "\n".join(lines)
