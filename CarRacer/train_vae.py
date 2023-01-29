"""
    Trains the convolutional variational autoencoder from the scenarios
    collected in the random rollouts.
"""

import os
from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.utils.data
import config
from pathlib import Path
from CarRacer import vae
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataset import ReplayDataset, get_dataloaders


def test_single_batch(v: Optional[vae.V] = None):

    if v is None:
        v = vae.V(**config.VAE)

    v.to(config.DEVICE)

    # test_dataset = dataset.VaeDataset(config.VAE_DATASET_DIR / "test")
    # test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)
    train_d, test_d = get_dataloaders(shuffle_test=True)

    batch, _ = next(iter(test_d))
    batch = batch.to(config.DEVICE)
    recon, mu, logvar = v(batch)
    loss = vae.vae_loss(recon, batch, mu, logvar)

    # plot the original and reconstructed frames
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        ax[0, i].imshow(batch[i].cpu().permute(1, 2, 0))  # type: ignore
        ax[1, i].imshow(recon[i].cpu().detach().permute(1, 2, 0))  #type: ignore
    plt.show()


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
    v: vae.V,
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
    pbar = tqdm(total=len(dataloader.dataset))  #type: ignore

    # set vae to correct mode
    v.train() if train else v.eval()

    # start of epoch
    for i, (batch, replay_indices) in enumerate(dataloader):
        # move to CUDA
        batch = batch.to(config.DEVICE)

        # forward pass and backpropagation
        if train and optimizer:
            recon, mu, logvar = v(batch)
            loss = vae.vae_loss(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                recon, mu, logvar = v(batch)
                loss = vae.vae_loss(recon, batch, mu, logvar)

        # metrics
        avg_loss: float = loss_tracker(loss.item())

        # update prograss bar
        max_replay_index = torch.max(replay_indices).item()
        pbar.n = max_replay_index
        pbar.set_description_str(
            f"{'Train' if train else 'Test'} "
            f"epoch{(' ' + str(epoch_index+1)) if train else ''}: "
            f"Loss: {avg_loss:.2f}")
        pbar.refresh()
        pbar.update()

    return avg_loss


# train the vae on the replay fils
def train_vae(
    v: Optional[vae.V] = None,
    num_epochs: int = 1,
    batch_size_train: int = 32,
    display: bool = True,
):
    """ Train the VAE on the replay files

    Args:
        num_epochs (int, optional): Number of epochs to train for. Defaults to 1.
        batch_size_train (int, optional): Batch size for training. Defaults to 32.
    """

    if v is None:
        v = vae.V(**config.VAE).to(config.DEVICE)

    optimizer = torch.optim.Adam(v.parameters(), lr=1e-3)
    train_d, test_d = get_dataloaders(batch_size_train=batch_size_train)

    for epoch in range(num_epochs):
        best_loss: float = float("inf")
        # train epoch
        train_loss = run_epoch(v, train_d, optimizer, epoch, train=True)
        # test epoch
        test_loss = run_epoch(v, test_d, epoch_index=epoch, train=False)
        print(f"Epoch {epoch + 1} - "
              f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
        # save model checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(v.state_dict(), config.CKP_DIR / "vae.pt")

    if display:
        test_single_batch(v)


# function to test the vae
def test_vae(v: vae.V, display: bool = True):
    _, test_d = get_dataloaders(num_workers=1)
    loss = run_epoch(v, test_d, train=False)
    print(f"Test loss: {loss:.4f}")

    if display:
        test_single_batch(v)

    return loss


def save_hidden_states():

    v = vae.load_vae()
    v.eval()
    v.to(config.DEVICE)
    replays = sorted(config.REPLAY_DIR.glob("*.npz"), key=os.path.basename)

    for replay in tqdm(replays,
                       desc="Converting frames to latent space...",
                       total=len(replays)):
        data = np.load(replay)
        obs = data["obs"]

        hidden_states = []
        for image in obs:
            image_tensor = (torch.tensor(image, dtype=torch.float32) /
                            255).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                out = v.encode(image_tensor)
            hidden_states.append(out[0].cpu().numpy())

        hidden_states = np.array(hidden_states, dtype=np.float32)
        name = replay.stem
        np.savez_compressed(config.Z_DIR / f"{name}.npz", z=hidden_states)


if __name__ == "__main__":
    v = vae.load_vae()
    test_single_batch(v)

# # function to load vae from checkpoint
# def load_vae() -> vae.V:
#     """ Load the VAE from the checkpoint

#     Returns:
#         vae: the VAE
#     """
#     v = vae.V(**config.VAE).to(config.DEVICE)
#     v.load_state_dict(torch.load(config.CKP_DIR / "vae.pt"))
#     return v

# train_vae(batch_size_train=100, num_epochs=2)
# v = vae.load_vae()
# test_vae(v)
# test_single_batch(v)

# save_hidden_states()
# dataloader = get_dataloader()
# len(dataloader)
# for batch, indices in dataloader:
#     for image in batch:
#         plt.imshow(image.permute(1, 2, 0))
#         plt.show()
