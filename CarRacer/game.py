"""
    Implementation of game-related stuff
"""

import math
# from time import time
import time
import torch.multiprocessing as multiprocessing
from typing import Callable, Optional
import numpy as np
from gym.envs.box2d.car_racing import CarRacing
from gym.spaces.box import Box
import pygame
import gym
from PIL import Image
import os
import vae
import memory
import controller
import config
import torch
import random
import timeit

from config import IMWIDTH, IMHEIGHT


class ResizeFrame(gym.ObservationWrapper):
    """
    Environment wrapper which alters the frame images.
    Removes the bottom section containing score and other info and 
    resizes the frame to 64x64
    """

    def __init__(self, env: CarRacing):
        super().__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(IMWIDTH, IMHEIGHT, 3))

    def observation(self, frame: np.ndarray) -> np.ndarray:
        # cut the bottom section of the image containing the score
        frame = frame[0:84, :, :]
        # resize image with pillow to 64x64
        frame = np.array(Image.fromarray(frame).resize((IMWIDTH, IMHEIGHT)))
        return frame


def play_rollout(strategy: Callable, seed: int = -1, render: bool = False):
    """Plays a rollout of a game using the given strategy to choose the actions

    Args:
        strategy (Callable): The strategy function. Takes action vector and obs
        seed (int, optional): Random seed. Defaults to -1.
        render (bool, optional): Whether to render the game. Defaults to False.
    """
    # env = CarRacing(render_mode="human" if render else "rgb_array")
    env = ResizeFrame(CarRacing(render_mode="human" if render else "rgb_array"))

    if seed >= 0:
        env.seed(seed)
    isopen = True
    if render:
        env.render()
    obs, info = env.reset()
    actions = np.array([0.0, 0.0, 0.0])
    total_reward = 0.0
    steps = 0
    log_probs = []
    rewards = []
    while steps < config.MAX_STEPS:
        strategy_out = strategy(obs, actions)

        if isinstance(strategy_out, tuple):
            actions, log_prob = strategy_out
            log_probs.append(log_prob)

        obs, reward, terminated, truncated, _ = env.step(actions)  #type: ignore
        rewards.append(reward)
        total_reward += reward
        steps += 1
        if render:
            isopen = env.render()
        if terminated or truncated or isopen is False:
            break
    env.close()
    return rewards, log_probs


def human_strategy(obs, a):
    """Strategy that asks for human input to choose the actions to take.

    Args:
        a (np.array): the previous state of the action vector
        obs (np.array): the observation

    Returns:
        np.array: the action vector
    """
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[2] = +0.8
            if event.key == pygame.K_RETURN:
                global restart
                restart = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[2] = 0
    return a


import random


def random_strategy(obs, a):
    steer = random.uniform(-1, 1)
    gas = random.uniform(0, 1)
    brake = random.uniform(0, 1)
    gas = (gas + 1) / 2

    return np.array([steer, gas, brake])


def fullblast_strategy(obs, a):
    """
    GO VROOM
    """
    return np.array([0.0, 1.0, 0.0])


class RandomAgent:

    def __init__(self, device: torch.device = config.DEVICE):

        self.device = device
        self.v = vae.V(**config.VAE).to(self.device)
        self.m = memory.M(**config.LSTM, device=self.device).to(self.device)
        self.c = controller.C(**config.CONTROLLER).to(self.device)

        self.v.eval()
        self.m.eval()
        self.c.eval()

    def find_move(self, obs, a) -> np.ndarray:
        with torch.no_grad():
            # convert frame to tensor
            # obs_tensor = (torch.tensor(obs, dtype=torch.float32) / 255).reshape(
            #     1, 64, 64, 3).permute(0, 3, 1, 2).to(self.device)
            obs = (torch.from_numpy(obs).float() / 255).reshape(
                1, config.IMHEIGHT, config.IMWIDTH,
                3).permute(0, 3, 1, 2).to(self.device)

            # use vae to get latent encoding for the frame
            z, _, _ = self.v.encode(obs)

            # concatenate z with current lstm hidden state
            if self.m.hidden is None:
                self.m.init_hidden(1)
            hidden, _ = self.m.hidden
            z_h = (torch.cat((z.squeeze(), hidden.squeeze()), dim=-1))

            # use controller to choose the best action
            # z_h = torch.rand(1, 32 + 256).to(self.device)
            action = self.c.predict(z_h)
            # action[..., 1] = action[..., 1] * 0 + 1
            action[..., 2] = action[..., 2] * 0

            ## Update the hidden state and cell of the LSTM
            z_a = torch.cat((z.squeeze(), action), dim=-1)
            _ = self.m(z_a.unsqueeze(0), use_hidden=True)

        return action.squeeze().detach().cpu().numpy()


def save_episodes(num_episodes: int = 1,
                  render: bool = False,
                  device: torch.device = config.DEVICE,
                  update_every: int = 1,
                  process_name: Optional[str] = None,
                  fake_run: bool = True,
                  use_network: bool = False):
    """ Plays a number of episodes with a random policy and saves the frames and
        actions to a file.
    """

    if process_name is None:
        process_name = os.getpid()  #type: ignore

    # set up directory to save the episodes
    save_dir = config.REPLAY_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = ResizeFrame(CarRacing(render_mode="human" if render else "rgb_array"))

    for episode in range(num_episodes):
        # instantiate the agent and environment
        if use_network:
            agent = RandomAgent(device=device)

        # initialize seed and savefile
        seed = random.randint(0, 2**31 - 1)
        save_file = save_dir / f"{seed}.npz"
        np.random.seed(seed)
        obs, info = env.reset(seed=seed)
        torch.manual_seed(seed)

        # initialize lists of stuff to save
        record_obs = []
        record_actions = []
        record_reward = []

        # start rollout
        isopen = True
        if render:
            env.render()
        actions = np.array([0.0, 0.0, 0.0])
        total_reward = 0.0
        steps = 0
        while steps < config.MAX_STEPS_REPLAY:

            # CHOOSE ACTION
            if use_network:
                actions = agent.find_move(obs, actions)  #type: ignore
            else:
                actions = random_strategy(obs, actions)

            # save frame and chosen action
            record_obs.append(obs)
            record_actions.append(actions)

            # perform action
            obs, r, terminated, truncated, _ = env.step(actions)  #type: ignore

            # save reward
            record_reward.append(r)
            total_reward += r
            steps += 1
            if render:
                isopen = env.render()
            if terminated or truncated or isopen is False:
                break

        # end rollout
        # print(f"dead at {steps+1}")
        # print(f"total reward: {total_reward}")
        if (episode + 1) % update_every == 0:
            print(f"{process_name}: I just finished episode {episode+1}")

        record_obs = np.array(record_obs, dtype=np.uint8)
        record_actions = np.array(record_actions, dtype=np.float16)
        record_reward = np.array(record_reward, dtype=np.float16)

        if not fake_run:
            np.savez_compressed(
                save_file,
                obs=record_obs,
                action=record_actions,
                reward=record_reward,
            )

        torch.cuda.empty_cache()

    # end simulation
    env.close()
    print(f"{process_name}: I'm done!")
    return


class VAECGame(multiprocessing.Process):
    """
    Run the game in a new process and save the outcome.
    This class was taken from https://github.com/dylandjian/retro-contest-sonic
    and modified to suit my needs.
    """

    def __init__(self, process_id, controller, result_queue, render_mode: str):
        super(VAECGame, self).__init__()
        self.process_id = process_id
        self.controller = controller
        self.result_queue = result_queue
        self.convert = {0: 0, 1: 5, 2: 6, 3: 7}
        self.render_mode: str = render_mode

    def _convert(self, actions):
        """ Convert predicted action into an environment action 
            This is now done in the controller
        """

        actions[..., 0] = actions[..., 0] * 2 - 1
        actions[1] = (actions[1] + 1.0) / 2.0
        actions[2] = torch.clip(actions[2], 0, 1)

        return actions

    def run(self):
        self.rungame()

    def rungame(self):
        """ Called by process.start(). We instantiate the neural network 
            directly inside the subprocess to save GPU memory.
        """

        # create network
        self.vae = vae.load_vae()
        self.lstm = memory.load_memory()
        self.controller.to(config.DEVICE)
        self.vae.eval()
        self.lstm.eval()
        self.controller.eval()

        final_reward = []
        start_time = timeit.default_timer()
        env = ResizeFrame(CarRacing(render_mode=self.render_mode))

        for _ in range(config.REPEAT_ROLLOUT_TEST):
            obs, info = env.reset()
            done = False
            total_reward = 0
            total_steps = 0
            current_rewards = []

            for _ in range(config.MAX_STEPS_TEST):
                with torch.no_grad():

                    if self.lstm.hidden is None:
                        self.lstm.init_hidden(1)

                    ## Reset the hidden state once we have seen SEQUENCE number of frames
                    if total_steps % 100 == 0:
                        self.lstm.init_hidden(1)

                    lstm_hidden, _ = self.lstm.hidden

                    ## Predict the latent representation of the current frame
                    obs = (torch.from_numpy(obs).float() / 255).reshape(
                        1, config.IMHEIGHT, config.IMWIDTH,
                        3).permute(0, 3, 1, 2).to(config.DEVICE)
                    z, _, _ = self.vae.encode(obs)

                    ## Use the latent representation and the hidden state and cell of
                    ## the LSTM to predict an action vector
                    z_h = torch.cat((z.squeeze(), lstm_hidden.squeeze()),
                                    dim=-1)
                    # actions = self.controller(z_h)
                    # final_action = self._convert(actions)
                    final_action = self.controller.predict(z_h)

                    env_action = final_action.squeeze().detach().cpu().numpy()
                    obs, reward, terminated, truncated, _ = env.step(env_action)
                    if terminated or truncated:
                        break

                    ## Update the hidden state and cell of the LSTM
                    z_a = torch.cat((z.squeeze(), final_action), dim=-1)

                    _ = self.lstm(z_a.unsqueeze(0), use_hidden=True)

                total_reward += reward
                total_steps += 1

            final_reward.append(total_reward)

        final_time = timeit.default_timer() - start_time
        print("[{} / {}] Final mean reward: {}" \
                    .format(self.process_id + 1, config.POPULATION, np.mean(final_reward)))
        del self.vae
        del self.lstm
        env.close()
        result = {}
        result[self.process_id] = (np.mean(final_reward), final_time)
        self.result_queue.put(result)


def parallel_save_episodes_network(total_episodes: int,
                                   fake_run: bool = True,
                                   render: bool = True):
    """ Plays a lot of episodes in parallel and saves the runs to disk.
        Uses an untrained neural network to play the game.
        On my machine, a good utilization of the memory is achieved with 3 cuda
        processes, and 2 cpu processes.
    """
    running_processes = []
    half_episodes = int(math.ceil(total_episodes / 2))
    other_half = total_episodes - half_episodes
    total_processes = 3
    cuda_processes = 3
    cpu_processes = total_processes - cuda_processes
    print(f'Time: {time.strftime("%H:%M:%S", time.localtime())}')
    start = time.time()
    for p in range(total_processes):
        if p < cuda_processes:
            num_episodes = half_episodes // cuda_processes
            dev = "cuda:0"
            name = f"cuda:{p}"
        else:
            num_episodes = other_half // (cpu_processes)
            dev = "cpu"
            name = f"cpu:{p-cuda_processes}"

        process = torch.multiprocessing.Process(
            target=save_episodes,
            args=(num_episodes, render, torch.device(dev), 20, name, fake_run,
                  True),
            name=name,
        )

        process.start()
        print(f"Started process {name} with pid {process.pid}")

        running_processes.append(process)

    # wait for all processes to finish
    for process in running_processes:
        process.join()
        print(f"Process {process.name} ({process.pid}) collected.")

    end = time.time()
    print(f'Time: {time.strftime("%H:%M:%S", time.localtime())}')
    print(f'Time elapsed: {time.strftime("%H:%M:%S", time.gmtime(end-start))}')


def parallel_save_episodes_random(fake_run: bool = True):
    """ Plays a lot of episodes in parallel and saves the runs to disk.
        Uses a random policy to play the game.
        On my machine, a good utilization of the memory is achieved with 3 cuda
        processes, and 2 cpu processes.
    """
    running_processes = []
    n_processes = 10
    n_episodes = 100
    for p in range(10):
        num_episodes = n_episodes // n_processes
        dev = "cpu"
        name = f"cpu:{p}"

        process = torch.multiprocessing.Process(
            target=save_episodes,
            args=(num_episodes, fake_run, torch.device(dev), 20, name, fake_run,
                  False),
            name=name,
        )

        process.start()
        print(f"Started process {name} with pid {process.pid}")

        running_processes.append(process)

    # wait for all processes to finish
    for process in running_processes:
        process.join()
        print(f"Process {process.name} ({process.pid}) collected.")


if __name__ == "__main__":
    parallel_save_episodes_network(total_episodes=5000,
                                   fake_run=False,
                                   render=False)
