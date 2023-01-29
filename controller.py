"""
    Implementation of the C Model, the controller.

    It is responsible for the choice of the action to take to maximize the 
    expected cumulative reward of the agent during a rollout.

    Is a simple single linear model that maps z[t] and h[t] directly to an 
    action a[t].

    a[t] = W[z[t], h[t]] + b
"""

from typing import Tuple
import torch
from torch import nn
from dataclasses import dataclass
from torch.nn import functional as F
from pathlib import Path


@dataclass(repr=False, unsafe_hash=True)
class C(nn.Module):

    #Hyperparameters
    input_dim: int
    num_actions: int

    def __post_init__(self):
        super().__init__()
        self.fc = nn.Linear(self.input_dim, self.num_actions, bias=True)

        # self.fc = nn.Linear(self.input_dim, 256, bias=True)
        # self.fc2 = nn.Linear(256, self.num_actions)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)

        # x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def save_controller(c: C, path: Path):
    torch.save(c.state_dict(), path)
