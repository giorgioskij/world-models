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
import config
from torch.nn import functional as F


@dataclass(repr=False, unsafe_hash=True)
class C(nn.Module):

    # def __init__(self, input_dim, num_actions):
    #     super().__init__()
    #     self.input_dim: int = input_dim
    #     self.num_actions: int = num_actions

    #Hyperparameters
    input_dim: int
    num_actions: int

    def __post_init__(self):
        super().__init__()
        self.fc = nn.Linear(self.input_dim, self.num_actions, bias=False)

    def convert(self, actions):
        """ Converts the output of the network to a game action.
        """
        actions[..., 0] = actions[..., 0] * 2 - 1
        actions[..., 1] = (actions[..., 1] + 1.0) / 2.0
        actions[..., 2] = torch.clip(actions[..., 2], 0, 1)

        return actions

    def forward(self, x):
        # action = torch.tanh(self.fc(x))
        # action[..., 1] = (action[..., 1] + 1.0) / 2
        # action[..., 2] = torch.clip(action[..., 2], 0, 1)
        x = torch.sigmoid(self.fc(x))
        return x

    def predict(self, x):
        """ Returns the action to take, given the state.
        """
        return self.convert(self.forward(x))


@dataclass(repr=False, unsafe_hash=True)
class deep_C(nn.Module):

    # Hyperparameters
    input_dim: int
    hidden_dim: int
    num_actions: int

    def __post_init__(self):
        super().__init__()

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.mu = nn.Linear(self.hidden_dim, self.num_actions)
        self.sigma = nn.Linear(self.hidden_dim, self.num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.mu(x)
        sigma = self.sigma(x)

        # limit sigma to be positive
        sigma = F.softplus(sigma)

        # limit mu in range [-1, 1]
        mu = torch.tanh(mu)

        return mu, sigma

    def act(self, state) -> Tuple[torch.Tensor, torch.Tensor]:

        mu, sigma = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()

        action[..., 1] = (action[..., 1] + 1.0) / 2

        return action, dist.log_prob(action)


def save_controller(c: C):
    torch.save(c.state_dict(), config.CKP_DIR / "controller.pt")


# if __name__ == "__main__":
#     c_args = config.CONTROLLER
#     controller = C(**config.CONTROLLER)

#     print(controller)
#     from torchinfo import summary

#     print(summary(controller, (4, 288)))
