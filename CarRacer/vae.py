"""
    Implementation of the V Model - a variational autoencoder
    Its goal is to learn an abstract, compressed representation of each observed
    input frame.

    It receives as input an image of a frame and outputs a vector z.
"""

from typing import Tuple
import torch
from torch import Tensor
from torch import nn
import config
from dataclasses import dataclass


class Reshape(nn.Module):
    """Trivial reshape nn module
    """

    def __init__(self, newdim):
        super().__init__()
        self.newdim = newdim

    def forward(self, x):
        return x.reshape(self.newdim)


@dataclass(repr=False, unsafe_hash=True)
class V(nn.Module):

    # Hyperparamters
    latent_dim: int
    input_dim: int

    def __post_init__(self):
        super().__init__()

        self.conv_out_dim: int = 2 * 2 * 256

        # Encoder
        self.encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Latent representation
        self.fc_mu = nn.Linear(self.conv_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.conv_out_dim, self.latent_dim)

        # Decoder
        self.decoder: nn.Module = nn.Sequential(
            nn.Linear(self.latent_dim, self.conv_out_dim),
            nn.ReLU(),
            Reshape((-1, self.conv_out_dim, 1, 1)),
            nn.ConvTranspose2d(self.conv_out_dim, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )

    def latent_sample(self, mu, logvar) -> Tensor:
        # Convert logvar to std
        std = (logvar * 0.5).exp()
        # Reparametrization trick
        z = torch.distributions.Normal(loc=mu, scale=std).rsample()
        return z

    def encode(self, obs) -> Tuple[Tensor, Tensor, Tensor]:
        # encode frame
        conv_out = self.encoder(obs)
        mu = self.fc_mu(conv_out)
        logvar = self.fc_logvar(conv_out)
        # sample z
        z = self.latent_sample(mu, logvar)
        return z, mu, logvar

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor]:
        z, mu, logvar = self.encode(x)
        # Reconstruct
        y = self.decoder(z)
        return y, mu, logvar

    def decode(self, x):
        return self.decoder(x)


def vae_loss(recon_x,
             x,
             mu: Tensor,
             logvar: Tensor,
             kl_tolerance: float = 0.5,
             latent_dim: int = 32):

    # reconstruction loss as in the original paper
    recon_loss = torch.square(x - recon_x).sum(dim=(1, 2, 3))
    recon_loss = torch.mean(recon_loss)

    # kl divergence as in the paper
    kl_loss = -0.5 * torch.sum(
        input=(1 + logvar - torch.square(mu) - torch.exp(logvar)), dim=1)
    kl_loss = torch.maximum(kl_loss, torch.tensor(kl_tolerance * latent_dim))
    kl_loss = torch.mean(kl_loss)

    return recon_loss + kl_loss * config.KL_WEIGHT

    return {"reconstruction": recon_loss, "kl": kl_loss}


# function to load vae from checkpoint
def load_vae() -> V:
    """ Load the VAE from the checkpoint

    Returns:
        vae: the VAE
    """
    v = V(**config.VAE).to(config.DEVICE)
    v.load_state_dict(torch.load(config.CKP_DIR / "vae.pt"))
    return v


if __name__ == "__main__":
    vae_args = config.VAE
    vae = V(**vae_args)

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f'{vae=}')
    print(f'{num_params=}')

    from torchinfo import summary

    summary(vae)