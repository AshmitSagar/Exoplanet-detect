"""
src/pipeline/AI/training/__init__.py

Training package for AstroNetCNN on Kepler NPZ data.

Public surface
──────────────
    from training.config     import TrainingConfig
    from training.metrics    import MetricAccumulator, accuracy, precision_recall_f1
    from training.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
    from training.trainer    import Trainer
"""

from .config     import TrainingConfig
from .metrics    import MetricAccumulator, accuracy, precision_recall_f1
from .checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from .trainer    import Trainer

__all__ = [
    "TrainingConfig",
    "MetricAccumulator",
    "accuracy",
    "precision_recall_f1",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "Trainer",
]
