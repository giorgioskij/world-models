import config as cfg
from typing import Optional
from implemented_games import Games
from vae import V
from memory import M
from controller import C
import torch
from torch.nn import functional as F
from enum import Enum, auto


class Strategy(Enum):
    RANDOM = 0
    MODEL = auto()


class Dreamer:

    def __init__(self,
                 game: Games,
                 strategy: Strategy,
                 device: torch.device = cfg.DEVICE):
        self.strategy: Strategy = strategy
        self.device: torch.device = device
        self.game: Games = game
        self.num_actions: int = game.num_actions  #type:ignore

        self.memory: Optional[M] = None
        self.vae: Optional[V] = None
        self.controller: Optional[V] = None

    def load_models(self):
        self.load_vae()
        self.load_memory()
        self.load_controller()

    def load_vae(self):
        self.vae = V(**cfg.VAE)
        self.vae.load_state_dict(torch.load(cfg.CKP_DIR / "vae.pt"))
        self.vae.to(self.device)

    def load_memory(self):
        self.memory = M(**cfg.LSTM, input_size=cfg.Z_DIM + self.num_actions)
        self.memory.load_state_dict(torch.load(cfg.CKP_DIR / "memory.pt"))
        self.memory.to(self.device)

    def init_memory(self, batch_size: int):
        if self.memory is None:
            raise ModuleNotFoundError("Memory is not loaded")
        self.memory.init_hidden(batch_size)

    def load_controller(self, controller: Optional[C] = None):
        if controller is None:
            controller = C(**cfg.CONTROLLER, num_actions=self.num_actions)
            controller.load_state_dict(torch.load(cfg.CKP_DIR /
                                                  "controller.pt"))
        self.controller = controller  # type: ignore
        self.controller.to(self.device)

    def choose_action(self, obs):
        if self.strategy == Strategy.RANDOM:
            return self.choose_action_random()
        elif self.strategy == Strategy.MODEL:
            return self.choose_action_model(obs)
        else:
            raise NotImplementedError

    def choose_action_random(self):
        return self.game.sampleAction()

    def choose_action_model(self, obs):

        with torch.no_grad():

            if self.controller is None or self.memory is None or self.vae is None:
                raise ModuleNotFoundError("Some module is not loaded")

            # if lstm is not initialized, or every x steps initialize hidden state
            if self.memory.hidden is None:
                self.memory.init_hidden(1)
            memory_hidden, _ = self.memory.hidden

            # convert observation to tensor
            obs = (torch.from_numpy(obs).float() / 255).reshape(
                1, cfg.IMHEIGHT, cfg.IMWIDTH, 3).permute(0, 3, 1,
                                                         2).to(self.device)

            # encode observation
            z, _, _ = self.vae.encode(obs)

            # concatenate latent vector with lstm hidden state
            z_h = torch.cat((z.squeeze(), memory_hidden.squeeze()), dim=-1)

            # predict the action to take
            predicted_action = self.controller.predict(z_h)  # type:ignore

            # concatenate chosen action and latent vector to update memory
            action_one_hot = F.one_hot(predicted_action, num_classes=9)
            z_a = torch.cat((z.squeeze(), action_one_hot), dim=-1)

            # update memory
            _ = self.memory(z_a.unsqueeze(0), use_hidden=True)

            # return action
            return predicted_action.detach().cpu().numpy()
