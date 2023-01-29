""" 
    Train the memory model MRD-RNN on the random replay files to learn 
    how to predict the future based on the current state (z) and action (a).
"""

from game import frames_to_video
from implemented_games import Games
import config as cfg
import random
import time
from typing import Optional
import memory
import dataset
import torch
from torch.nn.utils import rnn as rnnutils
import mdn
import train_vae
import torch.utils.data
from train_vae import make_averager
from tqdm import tqdm


def run_epoch(
    model: memory.M,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_index: int = 0,
    train: bool = True,
):
    model = model.train() if train else model.eval()

    # set up metrics
    avg_loss: float = float("inf")
    loss_tracker = make_averager()
    pbar = tqdm(total=len(dataloader))

    for _, data in enumerate(dataloader):
        data = data.to(cfg.DEVICE)

        # create target distribution to compute loss
        unpacked_data, lengths = rnnutils.pad_packed_sequence(data)
        zero_vector = torch.zeros(1,
                                  unpacked_data.shape[1],
                                  unpacked_data.shape[2] - cfg.GAME.num_actions,
                                  device=cfg.DEVICE)
        target = torch.cat(
            (unpacked_data[1:, :, :-cfg.GAME.num_actions], zero_vector), dim=0)

        # forward pass and backpropagation
        if optimizer is not None:
            pi, sigma, mu = model(data)
            loss = mdn.mdn_loss(pi, sigma, mu, target, lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pi, sigma, mu = model(data)
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
    model: Optional[memory.M] = None,
    num_epochs: int = 1,
    batch_size_train: int = 32,
):

    if model is None:
        model = memory.M(**cfg.LSTM).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = dataset.LstmDataset()
    test_dataset = dataset.LstmDataset(test=True)
    train_d = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size_train,
                                          collate_fn=dataset.lstm_collate_fn)
    test_d = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=256,
                                         collate_fn=dataset.lstm_collate_fn)

    for epoch in range(num_epochs):
        best_loss: float = float('inf')
        # train epoch
        train_loss = run_epoch(model, train_d, optimizer, epoch, train=True)
        # test epoch
        test_loss = run_epoch(model, test_d, epoch_index=epoch, train=False)
        print(f"Epoch {epoch + 1} - "
              f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

        # save model checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), cfg.CKP_DIR / "memory.pt")


def test_memory(model: memory.M):
    test_dataset = dataset.LstmDataset(test=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, collate_fn=dataset.lstm_collate_fn)
    loss = run_epoch(model, test_dataloader, train=False)
    print(f"Test loss: {loss:.4f}")
    return loss


def simulate(m: Optional[memory.M] = None):

    if m is None:
        m = memory.M(**cfg.LSTM)
        m.load_state_dict(torch.load(cfg.CKP_DIR / "memory.pt"))

    m.to(cfg.DEVICE)
    m.eval()

    v = train_vae.load_vae()
    v.eval()
    v.to(cfg.DEVICE)

    # load some frames from a game and let the lstm predict the rest of the game
    # for now just use the first game
    # _, test_d = get_lstm_dataloaders(batch_size_test=1)
    test_dataset = dataset.LstmDataset(test=True)
    test_d = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=1,
                                         collate_fn=dataset.lstm_collate_fn)
    n_game = random.randint(0, len(test_d) - 2)
    it = iter(test_d)
    print(f"Game {n_game}")
    for _ in range(n_game):
        next(it)
    data = next(it)
    data = data.to(cfg.DEVICE)

    hidden_states, lengths = rnnutils.pad_packed_sequence(data)

    n_premoves = 50
    actions = hidden_states[:, :, -cfg.GAME.num_actions:]
    num_frames = hidden_states.shape[0]
    hidden_states = hidden_states[:n_premoves, :, :]

    # feed the first frames into the lstm
    for i in range(n_premoves, num_frames):
        with torch.no_grad():
            pi, sigma, mu = m(hidden_states)
            # print(pi.shape, sigma.shape, mu.shape)

        # sample from the distribution and append to frames
        new_frame = mdn.sample(pi[-1], sigma[-1], mu[-1])
        new_frame = torch.cat((new_frame, actions[i, :, :]), dim=-1)
        hidden_states = torch.cat((hidden_states, new_frame.unsqueeze(0)),
                                  dim=0)

    hidden_states = hidden_states[:, :, :-cfg.GAME.num_actions]
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    frames = v.decode(hidden_states)

    frames[n_premoves - 10:n_premoves, :, :] = torch.zeros_like(frames[0])

    # frames = frames.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()

    frames_to_video(frames, cfg.VIDEO_DIR / "simulation.avi")
    # show_game(frames, n_premoves)


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


# function to load memory from checkpoint
def load_memory(filename: str = "memory.pt") -> memory.M:
    m = memory.M(**cfg.LSTM).to(cfg.DEVICE)
    m.load_state_dict(torch.load(cfg.CKP_DIR / filename))
    return m


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    m = load_memory("memory.pt")
    # train_memory(m, num_epochs=10)
    # test_memory(m)
    # simulate(m)
    # m = load_memory()

    m = memory.M(**cfg.LSTM,).to(cfg.DEVICE)
    # test_memory(m)
    # train_memory(m, num_epochs=20)
    # simulate(m)
    # ...
