from pathlib import Path
import os

ROOT: Path = Path(__file__).parent.parent
os.chdir(ROOT)

from global_config import *

GAME_DIR: Path = ROOT / "CarRacer"
REPLAY_DIR: Path = GAME_DIR / "replays"
CKP_DIR: Path = GAME_DIR / "checkpoints"
VIDEO_DIR: Path = GAME_DIR / "videos"
Z_DIR: Path = GAME_DIR / "latent_states"

VAE = dict(
    latent_dim=Z_DIM,
    input_dim=64,
)
KL_WEIGHT: float = 1
NUM_ACTIONS = 3

LSTM = dict(
    sequence_len=1000,
    hidden_dim=lstm_hidden_dim,
    z_dim=Z_DIM,
    num_layers=1,
    n_gaussians=5,
    input_size=Z_DIM + NUM_ACTIONS,
)

DEEP_C = dict(
    input_dim=Z_DIM + lstm_hidden_dim,
    hidden_dim=100,
    num_actions=NUM_ACTIONS,
)
CONTROLLER = dict(
    input_dim=Z_DIM + lstm_hidden_dim,
    num_actions=NUM_ACTIONS,
)