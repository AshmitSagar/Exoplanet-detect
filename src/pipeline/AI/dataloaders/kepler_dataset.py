from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_MAP = {
    "PC": 0,     # Planet Candidate
    "AFP": 1,    # Astrophysical False Positive
    "NTP": 2,    # Non-Transiting Phenomenon
    "UNK": 3,    # Unknown
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class KeplerDataset(Dataset):
    """
    PyTorch Dataset for AstroNet Kepler NPZ files.

    Each sample returns:
        global_view : Tensor (2001,)
        local_view  : Tensor (201,)
        label        : Tensor ()
        metadata     : dict
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        self.files = sorted(self.data_dir.glob("*.npz"))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No NPZ files found in {self.data_dir}"
            )

        print(f"Loaded {len(self.files):,} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        sample = np.load(self.files[idx], allow_pickle=True)

        global_view = torch.tensor(
            sample["global_view"],
            dtype=torch.float32
        ).unsqueeze(0)

        local_view = torch.tensor(
            sample["local_view"],
            dtype=torch.float32
        ).unsqueeze(0)

        label_name = str(sample["label"])

        label = torch.tensor(
            LABEL_MAP[label_name],
            dtype=torch.long
        )

        metadata = {
            "kepid": int(sample["kepid"]),
            "planet_number": int(sample["planet_number"]),
            "period": float(sample["period"]),
            "label_name": label_name,
        }

        return {
            "global_view": global_view,
            "local_view": local_view,
            "label": label,
            "metadata": metadata,
        }