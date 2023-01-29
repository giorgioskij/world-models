import os
from typing import Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.nn import utils as nnutils
import config as cfg
from tqdm import tqdm
import torchvision as tv


class VaeDatasetLazy(torch.utils.data.IterableDataset):
    """ Dataset for the scenarios collected in the random rollouts. """

    def __init__(
        self,
        replay_dir: Path = cfg.REPLAY_DIR,
        shuffle: bool = True,
        seed: Optional[int] = None,
        test: bool = False,
        split_percent: float = 0.8,
        randomize: bool = False,
    ):
        self.replay_dir = replay_dir
        self.replays = sorted(replay_dir.glob("*.npz"), key=os.path.basename)
        self.shuffle: bool = shuffle

        # split the dataset
        num_train = int(len(self.replays) * split_percent)
        if test:
            self.replays = self.replays[num_train:]
            # self.shuffle = False
        else:
            self.replays = self.replays[:num_train]

        self.num_replays = len(self.replays)
        self.seed: Optional[int] = seed
        self.test: bool = test
        self.randomize: bool = randomize

        if self.randomize:
            self.generic_mask: torch.Tensor = self.get_generic_mask()

    def __len__(self):
        return self.num_replays

    # iter method to get a single scenario from the dataset
    # each time yields an image from the current file, when the file is done
    # it moves on to the next file
    def __iter__(self):
        for r_index, replay in enumerate(self.replays):
            try:
                data = np.load(replay)
                obs = data["observations"]
                actions = data["actions"]
            except:
                print(f"Error loading replay {replay}")
                continue
            if self.shuffle:
                if self.seed is not None:
                    np.random.seed(self.seed)
                indices = np.random.permutation(len(obs))
            else:
                indices = np.arange(len(obs))

            for obs_index in indices:
                image = obs[obs_index]
                image_tensor = (torch.tensor(image, dtype=torch.float32) /
                                255).permute(2, 0, 1)
                if self.randomize:
                    image_tensor = self.transform_vec(image_tensor)
                yield image_tensor

    def dump_dataset(self, dump_path: Path):

        global_index = 0
        for replay in tqdm(self.replays,
                           "Loading and dumping replays...",
                           total=len(self.replays)):
            try:
                loaded = np.load(replay)
                obs = loaded["observations"]
            except:
                print(f"Error loading replay {replay}")
                continue

            # convert observations into tensors
            for i in range(0, len(obs), 2):
                image = obs[i]
                # convert to tensor
                image_tensor = (torch.tensor(image, dtype=torch.float32) /
                                255).permute(2, 0, 1)
                # if necessary, apply random patches
                if self.randomize:
                    image_tensor = self.transform_vec(image_tensor)

                tv.utils.save_image(image_tensor,
                                    dump_path / f"{global_index:07}.png")
                global_index += 1
        return

    def get_generic_mask(self):
        intervals = [(5, 10), (15, 20), (25, 30), (34, 39), (44, 49), (54, 59)]
        generic_mask = torch.zeros(3, 64, 64)
        for range1 in intervals:
            for range2 in intervals:
                generic_mask[:, range1[0]:range1[1], range2[0]:range2[1]] = 1
        return generic_mask

    def transform(self, img):
        intervals = [(5, 10), (15, 20), (25, 30), (34, 39), (44, 49), (54, 59)]
        mask = torch.ones_like(img)
        for range1 in intervals:
            for range2 in intervals:
                dist1 = range1[1] - range1[0]
                dist2 = range2[1] - range2[0]
                mask[:, range1[0]:range1[1],
                     range2[0]:range2[1]] = torch.rand(3, dist1, dist2)
        masked = img * mask
        return masked

    def transform_vec(self, img):
        generic_mask = self.generic_mask  # 1s in the areas to randomize
        random_mask = torch.rand_like(generic_mask)  # all randoms

        # randoms in the areas to randomize, 0s elsewhere
        random_mask *= generic_mask

        # original, but 0s in the areas to randomize
        masked_image = img * (1 - generic_mask)

        # original, but randoms in the areas to randomize
        final_image = masked_image + random_mask
        return final_image


class VaeDataset(torch.utils.data.Dataset):

    def __init__(self, root: Path):
        self.root: Path = root
        self.image_paths = sorted(root.glob("*.png"), key=os.path.basename)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = tv.io.read_image(str(self.image_paths[idx]))
        return image / 255


class LstmDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        replay_dir: Path = cfg.REPLAY_DIR,
        z_dir: Path = cfg.Z_DIR,
        test: bool = False,
        split_percent: float = 0.8,
        max_length: int = cfg.MAX_STEPS,
    ):
        super().__init__()

        self.max_length: int = max_length
        self.replay_dir = replay_dir
        self.z_dir = z_dir
        self.replays = sorted(replay_dir.glob("*.npz"), key=os.path.basename)
        self.hidden_states = sorted(z_dir.glob("*.npz"), key=os.path.basename)

        if len(self.replays) != len(self.hidden_states):
            raise ValueError("Number of replays and hidden states do not match")

        # split the dataset
        num_train = int(len(self.replays) * split_percent)
        if test:
            self.replays = self.replays[num_train:]
            self.hidden_states = self.hidden_states[num_train:]
        else:
            self.replays = self.replays[:num_train]
            self.hidden_states = self.hidden_states[:num_train]

        self.test: bool = test

    def __len__(self):
        return len(self.replays)

    def __iter__(self):
        for _, (replay, z) in enumerate(zip(self.replays, self.hidden_states)):
            data = np.load(replay)
            actions = data["actions"]
            hidden_states = np.load(z)["z"]

            actions = torch.tensor(actions, dtype=torch.long)
            actions = F.one_hot(actions, num_classes=cfg.GAME.num_actions)
            hidden_states = torch.tensor(hidden_states,
                                         dtype=torch.float32).squeeze()

            yield hidden_states, actions


def lstm_collate_fn(data):
    hidden_states, actions = zip(*data)
    lengths = torch.tensor(list(map(len, hidden_states)))
    lengths_actions = torch.tensor(list(map(len, actions)))
    if not lengths.equal(lengths_actions):
        raise ValueError("Lengths of hidden states and actions do not match")

    # add a dummy tensor with maximum length to obtain max padding always
    # dummy_z = torch.zeros((cfg.MAX_STEPS, hidden_states[0].shape[-1]),
    #                       dtype=torch.float32)
    # dummy_a = torch.zeros((cfg.MAX_STEPS), dtype=torch.float32)
    # padded_z = nnutils.rnn.pad_sequence((*hidden_states, dummy_z))
    # padded_actions = nnutils.rnn.pad_sequence((*actions, dummy_a))

    padded_z = nnutils.rnn.pad_sequence((hidden_states))  #type:ignore
    padded_actions = nnutils.rnn.pad_sequence((actions))  #type:ignore

    # remove dummy
    # padded_z = padded_z[:, :-1]
    # padded_actions = padded_actions[:, :-1].unsqueeze(-1)  #type:ignore

    # concatenate hidden states and actions, because lstm takes a 35-dim input
    data = torch.cat((padded_z, padded_actions), dim=-1)

    packed_data = nnutils.rnn.pack_padded_sequence(data,
                                                   lengths,
                                                   enforce_sorted=False)

    return packed_data


# def get_lstm_dataloaders(
#     batch_size_train: int = 32,
#     batch_size_test: int = 256,
#     num_workers: int = 1,
# ):
#     dataset_train = LSTMReplayDataset()
#     train_d = torch.utils.data.DataLoader(dataset_train,
#                                           batch_size=batch_size_train,
#                                           num_workers=num_workers,
#                                           collate_fn=lstm_collate_fn)
#     dataset_test = LSTMReplayDataset()
#     test_d = torch.utils.data.DataLoader(dataset_test,
#                                          batch_size=batch_size_test,
#                                          num_workers=num_workers,
#                                          collate_fn=lstm_collate_fn)
#     return train_d, test_d

# def get_dataloaders(
#     batch_size_train: int = 32,
#     batch_size_test: int = 1024,
#     num_workers: int = 1,
#     shuffle_test: bool = False,
#     shuffle_train: bool = True,
# ):

#     dataset_train = ReplayDataset(shuffle=shuffle_train)
#     train_d = torch.utils.data.DataLoader(
#         dataset_train,
#         batch_size=batch_size_train,
#         num_workers=num_workers,
#     )
#     dataset_test = ReplayDataset(test=True, shuffle=shuffle_test)
#     test_d = torch.utils.data.DataLoader(
#         dataset_test,
#         batch_size=batch_size_test,
#         num_workers=num_workers,
#     )
#     return train_d, test_d


def randomize_images(imgs: torch.Tensor):
    intervals = [(5, 10), (15, 20), (25, 30), (34, 39), (44, 49), (54, 59)]
    mask = torch.ones_like(imgs)
    for range1 in intervals:
        for range2 in intervals:
            dist1 = range1[1] - range1[0]
            dist2 = range2[1] - range2[0]
            mask[:, :, range1[0]:range1[1],
                 range2[0]:range2[1]] = torch.rand(4, 3, dist1, dist2)
    masked = imgs * mask
    return masked


def vectorized_randomize(img: torch.Tensor):
    intervals = [(5, 10), (15, 20), (25, 30), (34, 39), (44, 49), (54, 59)]
    generic_mask = torch.ones(3, 64, 64)
    for range1 in intervals:
        for range2 in intervals:
            generic_mask[:, range1[0]:range1[1], range2[0]:range2[1]] = 0
    specific_mask = torch.rand_like(generic_mask)
    specific_mask += generic_mask
    specific_mask = specific_mask.clip(0, 1)

    import matplotlib.pyplot as plt
    plt.imshow(specific_mask.permute(1, 2, 0))
    plt.show()

    img = torch.zeros(3, 64, 64)
    img[0, ...] = 1
    # plt.imshow(img.permute(1,2,0))
    # plt.show()

    masked_image = img * generic_mask
    plt.imshow(masked_image.permute(1, 2, 0))
    plt.show()

    final_image_excess = masked_image + specific_mask
    negative_mask = -generic_mask
    final_image = final_image_excess + negative_mask
    plt.imshow(final_image.permute(1, 2, 0))
    plt.show()


# create lazy datasets, then call the dump method to materialize final datasets
def create_datasets():
    # train
    d = VaeDatasetLazy(replay_dir=cfg.REPLAY_DIR / "enduro",
                       shuffle=False,
                       test=False)
    d.dump_dataset(cfg.VAE_DATASET_DIR / "enduro" / "train")
    # test
    d = VaeDatasetLazy(replay_dir=cfg.REPLAY_DIR / "enduro",
                       shuffle=False,
                       test=True)
    d.dump_dataset(cfg.VAE_DATASET_DIR / "enduro" / "test")


if __name__ == "__main__":

    ...
