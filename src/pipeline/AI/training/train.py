"""
train.py — Entry point for AstroNetCNN training on Kepler NPZ data.

Usage
─────
    # Full training run with defaults:
    python src/pipeline/AI/training/train.py

    # Override individual hyper-parameters:
    python src/pipeline/AI/training/train.py \\
        --data_dir data/ai_ready/npz \\
        --output_dir data/models/run_01 \\
        --batch_size 64 \\
        --num_epochs 200 \\
        --learning_rate 5e-4 \\
        --lr_scheduler cosine \\
        --patience 20

    # Resume a previous run:
    python src/pipeline/AI/training/train.py \\
        --resume_from data/models/run_01/checkpoints/last_model.pth

    # Smoke test (small dataset, 2 epochs, gradient & checkpoint checks):
    python src/pipeline/AI/training/train.py --smoke_test

Flow
────
    1. Parse CLI → build TrainingConfig
    2. Seed RNG (Python, NumPy, PyTorch)
    3. Detect device (CUDA → AMP enabled)
    4. Load KeplerDataset  →  random_split (train / val)
    5. Compute class weights from training split
    6. Build DataLoaders
    7. Instantiate AstroNetCNN + AstroNetConfig
    8. Build AdamW optimizer + LR scheduler
    9. Optionally resume from checkpoint
    10. Run Trainer.fit()
    11. Print final metrics summary
    12. If --smoke_test: run sanity checks then exit

Design notes
────────────
• This file is deliberately explicit — each step is a labelled block so it is
  easy to read sequentially and to extend (e.g., add a Weights & Biases hook
  between steps 10 and 11 without touching anything else).

• All imports that could fail gracefully (matplotlib, etc.) are inside the
  relevant function so a missing optional dependency never prevents training
  from starting.

• The script can be imported as a module (e.g., from a notebook) by calling
  main() with a pre-built TrainingConfig.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

# ── Add src/pipeline to path so imports work regardless of CWD ───────────────
_HERE = Path(__file__).resolve().parent                       # .../AI/training
_AI   = _HERE.parent                                          # .../AI
_SRC  = _AI.parent.parent                                     # .../src
for _p in (_AI, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Project imports ───────────────────────────────────────────────────────────
from pipeline.AI.dataloaders.kepler_dataset import KeplerDataset, LABEL_MAP
from pipeline.AI.models.astronet_cnn import AstroNetCNN, AstroNetConfig

from .config     import TrainingConfig
from .metrics    import compute_class_weights
from .checkpoint import load_checkpoint
from .trainer    import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers = [
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops where possible (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_datasets(cfg: TrainingConfig):
    """
    Load KeplerDataset and split into train / val subsets.

    Returns:
        (train_dataset, val_dataset, all_train_labels)
        where all_train_labels is a LongTensor of training-split labels
        used to compute class weights.
    """
    full_dataset = KeplerDataset(cfg.data_dir)
    n_total = len(full_dataset)

    n_val   = max(1, int(n_total * cfg.val_fraction))
    n_train = n_total - n_val

    log.info("Dataset split:  total=%d  train=%d  val=%d", n_total, n_train, n_val)

    # Generator pinned to cfg.seed for reproducible splits across resume runs
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Collect training labels for class-weight computation
    # (we iterate the Subset indices rather than loading full tensors)
    train_labels: List[int] = []
    for idx in train_ds.indices:
        sample = full_dataset[idx]
        train_labels.append(sample["label"].item())
    all_train_labels = torch.tensor(train_labels, dtype=torch.long)

    return train_ds, val_ds, all_train_labels


def _build_dataloaders(
    train_ds   : torch.utils.data.Dataset,
    val_ds     : torch.utils.data.Dataset,
    cfg        : TrainingConfig,
    device     : torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    common_kwargs = dict(
        num_workers  = cfg.num_workers,
        pin_memory   = (device.type == "cuda"),
        # 'persistent_workers' requires num_workers > 0
        persistent_workers = (cfg.num_workers > 0),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size = cfg.batch_size,
        shuffle    = True,
        drop_last  = False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = cfg.batch_size * 2,   # no gradient → bigger batch is fine
        shuffle    = False,
        drop_last  = False,
        **common_kwargs,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model, optimiser, scheduler
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(cfg: TrainingConfig, device: torch.device) -> AstroNetCNN:
    model_cfg = AstroNetConfig(
        num_classes  = cfg.num_classes,
        dropout_rate = cfg.dropout_rate,
    )
    model = AstroNetCNN(model_cfg).to(device)
    breakdown = model.parameter_breakdown()
    log.info(
        "Model parameters:  global=%s  local=%s  classifier=%s  total=%s",
        f"{breakdown['global_branch']:,}",
        f"{breakdown['local_branch']:,}",
        f"{breakdown['classifier']:,}",
        f"{breakdown['total']:,}",
    )
    return model


def _build_optimizer_and_scheduler(
    model : nn.Module,
    cfg   : TrainingConfig,
) -> tuple[torch.optim.AdamW, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max  = cfg.num_epochs,
            eta_min = cfg.lr_eta_min,
        )
    elif cfg.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = cfg.lr_step_size,
            gamma     = cfg.lr_gamma,
        )
    else:
        raise ValueError(
            f"Unknown lr_scheduler '{cfg.lr_scheduler}'. "
            f"Choose 'cosine' or 'step'."
        )

    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test(cfg: TrainingConfig, device: torch.device) -> None:
    """
    Minimal sanity check that runs in seconds.

    Verifies:
      ✓ Forward pass produces correct output shape
      ✓ Loss is a finite scalar
      ✓ Gradients flow through all trainable parameters
      ✓ No NaN/Inf in any gradient
      ✓ Checkpoint can be saved and reloaded with identical outputs
    """
    print("\n" + "=" * 60)
    print("  Smoke test")
    print("=" * 60)

    torch.manual_seed(cfg.seed)
    B = 4
    C = cfg.num_classes

    # ── Dummy data matching KeplerDataset shapes ──────────────────────────────
    global_view = torch.randn(B, 1, 2001, device=device)
    local_view  = torch.randn(B, 1, 201,  device=device)
    labels      = torch.randint(0, C, (B,), device=device)

    model = _build_model(cfg, device)
    model.train()

    optimizer, _ = _build_optimizer_and_scheduler(model, cfg)
    criterion    = nn.CrossEntropyLoss()

    # ── Forward ────────────────────────────────────────────────────────────────
    logits = model(global_view, local_view)
    assert logits.shape == (B, C), \
        f"Expected logits shape ({B}, {C}), got {tuple(logits.shape)}"
    print(f"  [✓] Forward pass shape : {tuple(logits.shape)}")

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss = criterion(logits, labels)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"  [✓] Loss               : {loss.item():.4f}  (finite ✓)")

    # ── Backward ───────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()

    # Verify every trainable parameter has a gradient
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"

    # Check for NaN/Inf in gradients
    bad_grads = [n for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None
                 and not torch.isfinite(p.grad).all()]
    assert not bad_grads, f"NaN/Inf in gradients: {bad_grads}"
    print(f"  [✓] Gradients          : all finite, all present")

    optimizer.step()
    print(f"  [✓] Optimizer step     : OK")

    # ── Checkpoint round-trip ──────────────────────────────────────────────────
    import tempfile, os
    cfg.create_output_dirs()
    ckpt_path = cfg.checkpoint_dir / "_smoke_test.pth"

    state = {
        "epoch"         : 0,
        "model_state"   : model.state_dict(),
        "optim_state"   : optimizer.state_dict(),
        "sched_state"   : None,
        "scaler_state"  : {},
        "best_val_loss" : float("inf"),
        "config"        : {},
        "metrics"       : {},
    }

    from .checkpoint import save_checkpoint, load_checkpoint
    save_checkpoint(state, ckpt_path)

    # Fresh model, reload checkpoint, check outputs match
    model2 = _build_model(cfg, device)
    load_checkpoint(ckpt_path, model2, device=device)
    model2.eval()
    model.eval()

    with torch.no_grad():
        out1 = model(global_view, local_view)
        out2 = model2(global_view, local_view)

    assert torch.allclose(out1, out2, atol=1e-5), \
        "Outputs differ after checkpoint reload!"
    print(f"  [✓] Checkpoint save/reload : outputs match ✓")

    # Clean up smoke-test checkpoint
    try:
        ckpt_path.unlink()
    except OSError:
        pass

    print("=" * 60)
    print("  Smoke test PASSED")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: Optional[TrainingConfig] = None) -> None:
    """
    Full training entry point.  Can be called from code with a pre-built cfg,
    or run as __main__ where cfg is built from CLI arguments.
    """
    # ── Step 1: Config ────────────────────────────────────────────────────────
    if cfg is None:
        cfg = _parse_args()

    # ── Step 2: Output directories & logging ──────────────────────────────────
    cfg.create_output_dirs()
    _setup_logging(cfg.log_dir)
    log.info("TrainingConfig:\n%s", cfg)

    # ── Step 3: Device + AMP ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cfg.mixed_precision = True
        cfg.pin_memory      = True
        log.info("CUDA detected → device=%s  AMP=True", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        cfg.mixed_precision = False
        log.info("CUDA not available → device=cpu  AMP=False")

    # ── Step 4: Seed ──────────────────────────────────────────────────────────
    _seed_everything(cfg.seed)

    # ── Step 5: Smoke test (optional, exits after) ────────────────────────────
    if getattr(cfg, "_smoke_test", False):
        _smoke_test(cfg, device)
        return

    # ── Step 6: Dataset & split ───────────────────────────────────────────────
    t0 = time.time()
    train_ds, val_ds, train_labels = _build_datasets(cfg)
    log.info("Dataset loaded in %.1f s", time.time() - t0)

    # ── Step 7: Class weights ─────────────────────────────────────────────────
    if cfg.use_class_weights:
        weights = compute_class_weights(train_labels, cfg.num_classes).to(device)
        log.info("Class weights (balanced): %s", weights.tolist())
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    # ── Step 8: DataLoaders ───────────────────────────────────────────────────
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, cfg, device)
    log.info(
        "DataLoaders:  train_batches=%d  val_batches=%d  num_workers=%d",
        len(train_loader), len(val_loader), cfg.num_workers,
    )

    # ── Step 9: Model ─────────────────────────────────────────────────────────
    model = _build_model(cfg, device)

    # ── Step 10: Optimizer + scheduler ────────────────────────────────────────
    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg)

    # ── Step 11: Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    if cfg.resume_from is not None:
        ckpt = load_checkpoint(
            cfg.resume_from, model, optimizer, scheduler, device=device
        )
        start_epoch = ckpt.get("epoch", 0)
        log.info("Resuming from epoch %d", start_epoch)

    # ── Step 12: Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        config       = cfg,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        device       = device,
        criterion    = criterion,
    )

    # ── Step 13: Train ────────────────────────────────────────────────────────
    t_start = time.time()
    history = trainer.fit(start_epoch=start_epoch)
    elapsed = time.time() - t_start

    # ── Step 14: Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Training complete")
    print("=" * 65)
    print(f"  Wall time        : {elapsed / 60:.1f} min")
    print(f"  Best epoch       : {trainer.checkpoint_manager.best_epoch}")
    print(f"  Best val loss    : {trainer.checkpoint_manager.best_val_loss:.4f}")
    if history["val_acc"]:
        best_acc_idx = int(np.argmax(history["val_acc"]))
        print(f"  Best val acc     : {history['val_acc'][best_acc_idx]:.4f}"
              f"  (epoch {best_acc_idx + 1})")
    if history["val_f1"]:
        best_f1_idx = int(np.argmax(history["val_f1"]))
        print(f"  Best val F1      : {history['val_f1'][best_f1_idx]:.4f}"
              f"  (epoch {best_f1_idx + 1})")
    print(f"  Best model       : {trainer.checkpoint_manager.best_model_path}")
    print(f"  Last model       : {trainer.checkpoint_manager.last_model_path}")
    print(f"  CSV log          : {cfg.log_dir / 'train_log.csv'}")
    print(f"  Plots            : {cfg.plot_dir}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainingConfig:
    """Build a TrainingConfig from CLI arguments, falling back to defaults."""
    defaults = TrainingConfig()

    p = argparse.ArgumentParser(
        description="Train AstroNetCNN on Kepler NPZ data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",     default=str(defaults.data_dir),
                   help="Directory containing .npz samples")
    p.add_argument("--output_dir",   default=str(defaults.output_dir),
                   help="Root output directory (checkpoints/logs/plots)")
    p.add_argument("--val_fraction", default=defaults.val_fraction, type=float,
                   help="Fraction of dataset for validation")
    p.add_argument("--seed",         default=defaults.seed, type=int)

    # ── DataLoader ────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",   default=defaults.batch_size,   type=int)
    p.add_argument("--num_workers",  default=defaults.num_workers,  type=int)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--learning_rate", default=defaults.learning_rate, type=float)
    p.add_argument("--weight_decay",  default=defaults.weight_decay,  type=float)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    p.add_argument("--lr_scheduler",  default=defaults.lr_scheduler,
                   choices=["cosine", "step"])
    p.add_argument("--lr_step_size",  default=defaults.lr_step_size,  type=int)
    p.add_argument("--lr_gamma",      default=defaults.lr_gamma,      type=float)
    p.add_argument("--lr_eta_min",    default=defaults.lr_eta_min,    type=float)

    # ── Training loop ─────────────────────────────────────────────────────────
    p.add_argument("--num_epochs",         default=defaults.num_epochs,         type=int)
    p.add_argument("--no_class_weights",   action="store_true",
                   help="Disable inverse-frequency class weighting")

    # ── Early stopping ────────────────────────────────────────────────────────
    p.add_argument("--patience",    default=defaults.patience,    type=int)
    p.add_argument("--min_delta",   default=defaults.min_delta,   type=float)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    p.add_argument("--save_every",  default=defaults.save_every,  type=int,
                   help="Save numbered snapshot every N epochs (0=disabled)")
    p.add_argument("--resume_from", default=None,
                   help="Path to checkpoint .pth file to resume from")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--dropout_rate", default=defaults.dropout_rate, type=float)

    # ── Smoke test ────────────────────────────────────────────────────────────
    p.add_argument("--smoke_test", action="store_true",
                   help="Run a quick sanity check then exit")

    args = p.parse_args()

    cfg = TrainingConfig(
        data_dir          = Path(args.data_dir),
        output_dir        = Path(args.output_dir),
        val_fraction      = args.val_fraction,
        seed              = args.seed,
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        learning_rate     = args.learning_rate,
        weight_decay      = args.weight_decay,
        lr_scheduler      = args.lr_scheduler,
        lr_step_size      = args.lr_step_size,
        lr_gamma          = args.lr_gamma,
        lr_eta_min        = args.lr_eta_min,
        num_epochs        = args.num_epochs,
        use_class_weights = not args.no_class_weights,
        patience          = args.patience,
        min_delta         = args.min_delta,
        save_every        = args.save_every,
        resume_from       = Path(args.resume_from) if args.resume_from else None,
        dropout_rate      = args.dropout_rate,
    )
    # Attach smoke_test flag (not a dataclass field — avoids polluting config)
    cfg._smoke_test = args.smoke_test  # type: ignore[attr-defined]
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
