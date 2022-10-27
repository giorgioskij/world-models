from pathlib import Path
import os
import sys

ROOT: Path = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from global_config import *

GAME_DIR: Path = ROOT / "Procgen"
REPLAY_DIR: Path = GAME_DIR / "replays/dodgeball"
CKP_DIR: Path = GAME_DIR / "checkpoints/dodgeball"
VIDEO_DIR: Path = GAME_DIR / "videos"
Z_DIR: Path = GAME_DIR / "latent_states"

Z_DIM = 64
LSTM_DIM = 256
NUM_ACTIONS_CHASER = 9
NUM_ACTIONS_DODGEBALL = 6
VAE = dict(
    latent_dim=Z_DIM,
    input_dim=64,
)
KL_WEIGHT = 0.0001

LSTM = dict(
    sequence_len=1000,
    hidden_dim=LSTM_DIM,
    z_dim=Z_DIM,
    num_layers=1,
    n_gaussians=5,
)

CONTROLLER = dict(input_dim=Z_DIM + lstm_hidden_dim,)

import gym

gym.logger.setLevel(gym.logger.ERROR)