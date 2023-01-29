"""
    Implementation of the M Model, a Mixture-Density-Network combined with a 
    Recurrent Neural Network (MDN-RNN).

    Its goal is to predict the future: in particular, it predicts the future
    z vectors that V is expected to produce.

    It will model P(z[t+1] | a[t], z[t], h[t])

    Receives as input a vector z and an action taken, and outputs a probability
    distribution for the next z.
"""

from typing import Tuple
from unicodedata import bidirectional
from torch import nn
from torch.nn import functional as F
import torch
from dataclasses import dataclass
import mdn
import global_config as cfg


@dataclass(repr=False, unsafe_hash=True)
class M(nn.Module):

    # Hyperparameters
    sequence_len: int
    hidden_dim: int
    z_dim: int
    num_layers: int
    n_gaussians: int
    input_size: int
    device: torch.device = cfg.DEVICE

    def __post_init__(self):
        super().__init__()

        # self.hidden = self.init_hidden(self.sequence_len)
        self.hidden: Tuple[torch.Tensor, torch.Tensor] = None  #type:ignore

        # Encoding
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers)

        # MDN
        self.mdn = mdn.MDN(self.hidden_dim, self.z_dim, self.n_gaussians)

        # Output
        # self.z_pi = nn.Linear(self.hidden_dim, self.n_gaussians * self.z_dim)
        # self.z_sigma = nn.Linear(self.hidden_dim, self.n_gaussians * self.z_dim)
        # self.z_mu = nn.Linear(self.hidden_dim, self.n_gaussians * self.z_dim)

        # self.fc = nn.Linear(self.hidden_dim,
        #                     self.n_gaussians * self.z_dim * 3,
        #                     bias=True)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers,
                             batch_size,
                             self.hidden_dim,
                             device=self.device)

        cell = torch.zeros(self.num_layers,
                           batch_size,
                           self.hidden_dim,
                           device=self.device)

        self.hidden = (hidden, cell)

    def update_hidden(self, hidden, cell):
        self.hidden = (hidden, cell)

    def forward(self, x, use_hidden=False):
        """ forward pass through the network updating the hidden state

        Args:
            x (torch.Tensor): input tensor of shape 
                (sequence_len, batch_size, latent_dim + action_dim)
        """
        self.lstm.flatten_parameters()

        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            batch_size = x.batch_sizes[0]
        else:
            batch_size = x.shape[1]

        # use last hidden state
        if self.hidden is None:
            self.init_hidden(batch_size)
        hidden, cell = self.hidden

        if use_hidden:
            # squeeze because if input is unbatched, we want to remove the batch dimension
            z, (new_hidden,
                new_cell) = self.lstm(x, (hidden.squeeze(1), cell.squeeze(1)))
            self.update_hidden(new_hidden, new_cell)
        else:
            z, _ = self.lstm(x)

        if isinstance(z, torch.nn.utils.rnn.PackedSequence):
            z, lengths = torch.nn.utils.rnn.pad_packed_sequence(z)

        # pass through mixture density network
        out = self.mdn(z)
        return out


# def mdn_loss(out_pi, out_sigma, out_mu, y):
#     epsilon = 1e-6
#     sequence_len = config.LSTM["sequence_len"]
#     z_dim = config.LSTM["z_dim"]

#     y = y.view(-1, sequence_len, z_dim + 1)
#     result = torch.distributions.Normal(loc=out_mu, scale=out_sigma)
#     result = torch.exp(result.log_prob(y))
#     result = torch.sum(result * out_pi, dim=2)
#     result = -torch.log(epsilon + result)
#     return torch.mean(result)
