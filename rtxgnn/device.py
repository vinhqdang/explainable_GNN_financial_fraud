"""Device selection: set RTX_DEVICE=cuda to train on a GPU (default: cpu)."""
import os

import torch

DEVICE = torch.device(os.environ.get("RTX_DEVICE", "cpu"))


def to_dev(data):
    return data.to(DEVICE) if DEVICE.type != "cpu" else data
