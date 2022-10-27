"""
    Holds every configuration variable that is shared between all subprojects.
    Each subpackage can define its own config.py file that imports this one.
"""
import torch
from dataclasses import dataclass
from pathlib import Path

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available else "cpu")
# DEVICE = torch.device("cpu")

MAX_STEPS: int = 1000
MAX_STEPS_REPLAY: int = 1000
IMWIDTH: int = 64
IMHEIGHT: int = 64

Z_DIM = 32
lstm_hidden_dim = 256

POPULATION: int = 33
PARALLEL: int = 3
REPEAT_ROLLOUT: int = 4
