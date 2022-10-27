""" 
    Train the memory model MRD-RNN on the random replay files to learn 
    how to predict the future based on the current state (z) and action (a).
"""

import random
import time
from typing import Optional
import memory
import config
from dataset import get_lstm_dataloaders
import torch
from torch.nn.utils import rnn as rnnutils
import mdn
import vae
import train_vae
import torch.utils.data
from train_vae import make_averager
from tqdm import tqdm

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dev = torch.device('cpu')


def run_epoch(
    m: memory.M,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_index: int = 0,
    train: bool = True,
):
    m.to(dev)
    m.train() if train else m.eval()

    # set up metrics
    avg_loss: float = float('inf')
    loss_tracker = make_averager()
    pbar = tqdm(total=len(dataloader))

    for i, data in enumerate(dataloader):
        data = data.to(dev)

        # create target distribution to compute loss
        unpacked_data, lengths = rnnutils.pad_packed_sequence(data)
        zero_vector = torch.zeros(1,
                                  unpacked_data.shape[1],
                                  unpacked_data.shape[2] - 3,
                                  device=dev)
        target = torch.cat((unpacked_data[1:, :, :-3], zero_vector), dim=0)

        # forward pass and backpropagation
        if optimizer is not None:
            pi, sigma, mu = m(data)
            loss = mdn.mdn_loss(pi, sigma, mu, target, lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pi, sigma, mu = m(data)
                loss = mdn.mdn_loss(pi, sigma, mu, target, lengths)

        # metrics
        avg_loss = loss_tracker(loss.item())

        # update progress bar
        pbar.set_description_str(
            f"{'Train' if train else 'Test'} "
            f"epoch{(' ' + str(epoch_index+1)) if train else ''}: "
            f"Loss: {avg_loss:.2f}")
        pbar.refresh()
        pbar.update()

    return avg_loss


def train_memory(
    m: Optional[memory.M] = None,
    num_epochs: int = 1,
    batch_size_train: int = 32,
):

    if m is None:
        m = memory.M(**config.LSTM)

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    train_d, test_d = get_lstm_dataloaders(batch_size_train=batch_size_train)

    for epoch in range(num_epochs):
        best_loss: float = float('inf')
        # train epoch
        train_loss = run_epoch(m, train_d, optimizer, epoch, train=True)
        # test epoch
        test_loss = run_epoch(m, test_d, epoch_index=epoch, train=False)
        print(f"Epoch {epoch + 1} - "
              f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

        # save model checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(m.state_dict(), config.CKP_DIR / "memory.pt")


def test_memory(m: memory.M):
    _, test_d = get_lstm_dataloaders()
    loss = run_epoch(m, test_d, train=False)
    print(f"Test loss: {loss:.4f}")
    return loss


def simulate(m: Optional[memory.M] = None):

    if m is None:
        m = memory.M(**config.LSTM)
        m.load_state_dict(torch.load(config.CKP_DIR / "memory.pt"))

    m.to(dev)
    m.eval()

    v = vae.load_vae()
    v.eval()
    v.to(dev)

    # load some frames from a game and let the lstm predict the rest of the game
    # for now just use the first game
    _, test_d = get_lstm_dataloaders(batch_size_test=1)
    n_game = random.randint(0, len(test_d) - 2)
    it = iter(test_d)
    print(f"Game {n_game}")
    for _ in range(n_game):
        next(it)
    data = next(it)
    data = data.to(dev)

    hidden_states, lengths = rnnutils.pad_packed_sequence(data)

    n_premoves = 50
    actions = hidden_states[:, :, -3:]
    num_frames = hidden_states.shape[0]
    hidden_states = hidden_states[:n_premoves, :, :]

    # feed the first ten frames into the lstm
    for i in range(n_premoves, num_frames):
        with torch.no_grad():
            pi, sigma, mu = m(hidden_states)
            # print(pi.shape, sigma.shape, mu.shape)

        # sample from the distribution and append to frames
        new_frame = mdn.sample(pi[-1], sigma[-1], mu[-1])
        new_frame = torch.cat((new_frame, actions[i, :, :]), dim=-1)
        hidden_states = torch.cat((hidden_states, new_frame.unsqueeze(0)),
                                  dim=0)

    hidden_states = hidden_states[:, :, :-3]
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    frames = v.decode(hidden_states)

    frames = frames.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
    show_game(frames, n_premoves)


from matplotlib import pyplot as plt


def show_game(frames, n_premoves):
    fig, ax = plt.subplots()

    ims = []
    for i in range(len(frames)):
        if i == n_premoves - 1:
            print("End of premoves")
        plt.imshow(frames[i])
        plt.show()
        time.sleep(0.1)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    m = memory.load_memory("memory.pt")
    # train_memory(m, num_epochs=10)
    # test_memory(m)
    simulate(m)
    # ...