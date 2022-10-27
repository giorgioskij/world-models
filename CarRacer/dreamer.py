from typing import Optional, Tuple
import torch
import config
import vae
import memory
import controller
import numpy as np


class Dreamer:

    def __init__(
        self,
        v: Optional[vae.V] = None,
        m: Optional[memory.M] = None,
        c: Optional[controller.C] = None,
        train_controller: bool = False,
        device: torch.device = config.DEVICE,
    ):

        self.train_controller: bool = train_controller
        self.device = device
        self.v: vae.V = v  #type:ignore
        self.m: memory.M = m  #type:ignore
        self.c: controller.deep_C = c  #type:ignore

        if self.v is None:
            self.v = vae.load_vae()
        if self.m is None:
            self.m = memory.load_memory()
        if self.c is None:
            self.c = controller.deep_C(**config.DEEP_C)

        self.v.to(self.device)
        self.m.to(self.device)
        self.c.to(self.device)

        if self.train_controller:
            self.c.train()
        else:
            self.c.eval()

    def find_move(self, obs, a) -> Tuple[np.ndarray, torch.Tensor]:
        """
            Takes an observation (numpy array of the frame) and runs it through
            all 3 models to find the best move. 
            The VAE and Memory are supposed to be trained already so they are 
            passed through with no gradient tracking.
        """

        # convert observation to tensor
        obs = (torch.from_numpy(obs).float() / 255).reshape(
            1, config.IMHEIGHT, config.IMWIDTH, 3).permute(0, 3, 1,
                                                           2).to(self.device)

        # use vae to get latent encoding for the frame
        with torch.no_grad():
            z, _, _ = self.v.encode(obs)

        # concatenate z with current lstm hidden state
        if self.m.hidden is None:
            self.m.init_hidden(1)

        lstm_hidden, _ = self.m.hidden
        z_h = torch.cat((z.squeeze(), lstm_hidden.squeeze()), dim=-1)

        # use controller to choose the best action
        if self.train_controller:
            action, log_probs = self.c.act(z_h)
        else:
            with torch.no_grad():
                action, log_probs = self.c.act(z_h)

        # concatenate z with action
        z_a = torch.cat((z.squeeze(), action), dim=-1)

        # go forward with the lstm
        # use unsqueeze to add a timestep dimension, input is still unbatched
        with torch.no_grad():
            self.m(z_a.unsqueeze(0), use_hidden=True)

        return action.squeeze().detach().cpu().numpy(), log_probs
