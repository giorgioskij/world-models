import random
import config as cfg
from vae import V
import torch
from typing import Callable, Optional, Tuple
import torch.utils.data
import dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import numpy as np

dev = cfg.DEVICE


def train_vae(num_epochs=1,
              randomize: bool = False,
              vae: Optional[V] = None,
              name: str = "vae.pt"):

    #  train_dataset = dataset.VaeDatasetLazy(randomize=randomize)
    #  test_dataset = dataset.VaeDatasetLazy(test=True, shuffle=False)

    # new datasets
    train_dataset = dataset.VaeDataset(cfg.VAE_DATASET_DIR / "train")
    test_dataset = dataset.VaeDataset(cfg.VAE_DATASET_DIR / "test")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=10)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1024,
                                 shuffle=False,
                                 num_workers=10)

    if vae is None:
        vae = V(**cfg.VAE)
    vae.to(dev)

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    print("Starting training")
    for epoch in range(num_epochs):
        best_loss = float("inf")

        # train epoch
        train_loss = run_epoch(vae,
                               train_dataloader,
                               optimizer,
                               epoch,
                               train=True)

        # test epoch
        test_loss = run_epoch(vae,
                              test_dataloader,
                              epoch_index=epoch,
                              train=False)

        print(f"Epoch {epoch+1}\tTrain loss: {train_loss:.4f}\t"
              f"Test loss: {test_loss:.4f}")

        test_single_batch(vae)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(vae.state_dict(), cfg.CKP_DIR / name)


def test_vae(vae: V, display: bool = True):
    # test_dataset = dataset.VaeDatasetLazy(test=True, shuffle=False)
    test_dataset = dataset.VaeDataset(cfg.VAE_DATASET_DIR / "test")
    test_dataloader = DataLoader(test_dataset, batch_size=2048)
    loss = run_epoch(vae, test_dataloader, train=False)
    print(f"Test loss: {loss:.4f}")
    if display:
        test_single_batch(vae)
    return loss


def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average

    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager

        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager


def run_epoch(
    vae: V,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_index: int = 0,
    train: bool = True,
) -> float:
    """ Runs a single epoch of training or testing
    
    Args:
        optimizer: the optimizer to use for training
        v: the model to train
        pbar: the progress bar to update
        dataloader: the dataloader to use
        epoch_index: the index of the current epoch
        train: whether to train or test the model
    """

    # set up metrics
    avg_loss = float("inf")
    loss_tracker = make_averager()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    # set vae to correct mode
    vae.train() if train else vae.eval()

    # start of epoch
    for i, batch in pbar:
        # move to CUDA
        batch = batch.to(cfg.DEVICE)

        # forward pass and backpropagation
        if train and optimizer:
            recon, mu, logvar = vae(batch)
            loss = vae.loss(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                recon, mu, logvar = vae(batch)
                loss = vae.loss(recon, batch, mu, logvar)

        # metrics
        avg_loss: float = loss_tracker(loss.item())

        # update prograss bar
        # max_replay_index = torch.max(replay_indices).item()
        # pbar.n = i
        pbar.set_description_str(
            f"{'Train' if train else 'Test'} "
            f"epoch{(' ' + str(epoch_index+1)) if train else ''}: "
            f"Loss: {avg_loss:.2f}")
        # pbar.refresh()
        # pbar.update()

    return avg_loss


def test_single_batch(vae: Optional[V] = None):

    if vae is None:
        vae = V(**cfg.VAE)

    vae.to(cfg.DEVICE)

    test_dataset = dataset.VaeDataset(cfg.VAE_DATASET_DIR / "test")
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

    batch = next(iter(test_dataloader))
    batch = batch.to(cfg.DEVICE)
    recon, mu, logvar = vae(batch)
    loss = vae.loss(recon, batch, mu, logvar)

    # plot the original and reconstructed frames
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        ax[0, i].imshow(batch[i].cpu().permute(1, 2, 0))  # type: ignore
        ax[1, i].imshow(recon[i].cpu().detach().permute(1, 2, 0))  #type: ignore
    plt.show()


def load_vae(name: str = "vae.pt") -> V:
    vae = V(**cfg.VAE).to(dev)
    vae.load_state_dict(torch.load(cfg.CKP_DIR / name))
    return vae


def save_hidden_states(vae: V):

    vae.eval()
    vae.to(dev)
    replays = sorted(cfg.REPLAY_DIR.glob("*.npz"), key=os.path.basename)

    for replay in tqdm(replays,
                       desc="Converting frames to latent space...",
                       total=len(replays)):
        data = np.load(replay)
        obs = data["observations"]

        hidden_states = []
        n_obs = len(obs)
        n_batches = n_obs // 256 + 1
        # for image in obs:
        #     image_tensor = (torch.tensor(image, dtype=torch.float32) /
        #                     255).permute(2, 0, 1).unsqueeze(0).to(dev)

        #     with torch.no_grad():
        #         out = vae.encode(image_tensor)
        #     hidden_states.append(out[0].cpu().numpy())
        for batch in np.array_split(obs, n_batches):
            batch_tensor = (torch.tensor(batch, dtype=torch.float32) /
                            255).permute(0, 3, 1, 2).to(dev)

            with torch.no_grad():
                encodings, _, _ = vae.encode(batch_tensor)
            hidden_states.extend(encodings.cpu().numpy())

        hidden_states = np.array(hidden_states, dtype=np.float32)
        name = replay.stem
        np.savez_compressed(cfg.Z_DIR / f"{name}.npz", z=hidden_states)


# I wrote a lot of wrappers to use VAEs from this repo: https://github.com/AntixK/PyTorch-VAE
# I kept a couple here, commented, in case they turn out useful

# from models import VanillaVAE

# class VanillaVaeWrapper(VanillaVAE):
#     """ Wrapper to try out the implementation of VanillaVAE from GitHub
#     """

#     def __init__(self, in_channels, latent_dim, hidden_dims=None, **kwargs):
#         super().__init__(in_channels, latent_dim, hidden_dims, **kwargs)

#     def forward(self, input: torch.Tensor,
#                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         out = super().forward(input)
#         return out[0], out[2], out[3]

#     def loss(self, recon, batch, mu, logvar):
#         out = super().loss_function(recon, batch, mu, logvar, M_N=cfg.KL_WEIGHT)
#         return out["loss"]

# from models import BetaVAE

# class BetaVaeWrapper(BetaVAE):
#     """ Wrapper to try out the implementation of VanillaVAE from GitHub
#     """

#     def __init__(self, in_channels, latent_dim, hidden_dims=[], **kwargs):
#         super().__init__(in_channels, latent_dim, hidden_dims, **kwargs)

#     def forward(self, input: torch.Tensor,
#                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         out = super().forward(input)
#         return out[0], out[2], out[3]

#     def loss(self, recon, batch, mu, logvar):
#         out = super().loss_function(recon, batch, mu, logvar, M_N=cfg.KL_WEIGHT)
#         return out["loss"]

if __name__ == "__main__":
    v = load_vae("vae_z32.pt")
    # v = V(**cfg.VAE).to(cfg.DEVICE)
    # v = BetaVaeWrapper(in_channels=3, latent_dim=128)

    # test_vae(v)
    # test_single_batch(v)
    # train_vae(vae=v, num_epochs=10, name="vae_z32.pt")
    save_hidden_states(v)
    # train_vae(num_epochs=100, randomize=False, vae=v, name="vae.pt")
    #  ...
