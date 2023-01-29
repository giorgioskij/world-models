from pathlib import Path
import os
import sys

ROOT: Path = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from global_config import *
from implemented_games import Games
"""
Ideally, to switch games one should be able to only change this global
variable. No warranty applies.
"""
GAME = Games.ENDURO

GAME_DIR: Path = ROOT / "Procgen"
REPLAY_DIR: Path = GAME_DIR / "replays/" / GAME.name.lower()
CKP_DIR: Path = GAME_DIR / "checkpoints/" / GAME.name.lower()
VIDEO_DIR: Path = ROOT / "out"
Z_DIR: Path = GAME_DIR / "latent_states" / GAME.name.lower()
VAE_DATASET_DIR = GAME_DIR / "vae_dataset" / GAME.name.lower()

Z_DIM = 32
LSTM_DIM = 256

VAE = dict(
    latent_dim=Z_DIM,
    input_dim=64,
)
KL_WEIGHT = 0.00025

LSTM = dict(
    sequence_len=1000,
    hidden_dim=LSTM_DIM,
    z_dim=Z_DIM,
    num_layers=1,
    n_gaussians=5,
    input_size=Z_DIM + GAME.num_actions,
)

CONTROLLER = dict(
    input_dim=Z_DIM + lstm_hidden_dim,
    # input_dim=Z_DIM,
    num_actions=GAME.num_actions,
)

import gym

gym.logger.setLevel(gym.logger.ERROR)

POPULATION: int = 40
PARALLEL: int = 8
REPEAT_ROLLOUT: int = 3
SIGMA_INIT: float = 4
