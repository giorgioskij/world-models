"""
    Implementation of game-related stuff.
    The game is Chaser, from the PROCGEN suite.
"""

import config as cfg
from implemented_games import Games
from pathlib import Path
import timeit
from typing import Optional, Union
import cv2
import gym
from PIL import Image
from gym.spaces.box import Box
import random
from agent import Dreamer, Strategy
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import controller


class DreamGame():

    def __init__(
        self,
        agent: Optional[Dreamer],
        render: bool = True,
        save_episodes: bool = False,
    ):

        self.agent: Optional[Dreamer] = agent
        self.render: bool = render
        self.save_episodes: bool = save_episodes
        # self.game: Games = game

    def play(self, seed: Optional[int] = None, num_episodes: int = 1):
        if self.agent is None:
            raise Exception("Agent is not initialized")

        seed = seed or random.randint(0, 2**31 - 1)
        np.random.seed(seed)
        torch.manual_seed(seed)

        total_steps = []
        total_rewards = []
        for _ in range(num_episodes):
            # initialize seed and savefile
            seed = random.randint(0, 2**31 - 1)
            if self.save_episodes:
                game_name = cfg.GAME.name.lower()
                # game_name = "chaser" if self.game == Games.CHASER else "dodgeball"
                save_file = cfg.REPLAY_DIR / game_name / f"{seed}.npz"

            # create environment
            env = gym.make(
                cfg.GAME.longname,
                render_mode="human" if self.render else "rgb_array",
                # distribution_mode="hard",
            )
            if cfg.GAME == Games.ENDURO:
                env = ResizeFrame(env)

            obs = env.reset()
            if cfg.GAME == Games.ENDURO:
                obs = obs[0]

            # initialize actions, rewards, observations arrays
            actions = []
            rewards = []
            observations = []

            # run episode
            done = False
            steps = 0
            total_reward = 0
            while not done and steps < cfg.MAX_STEPS:

                # every x steps, reset hidden state
                if self.agent.strategy == Strategy.MODEL and steps % 100 == 0:
                    self.agent.init_memory(1)
                    # self.agent.memory.init_hidden(1)

                # choose an action
                action = self.agent.choose_action(obs)

                # perform the action and get the reward and new observation
                out = env.step(action)  #type: ignore
                if cfg.GAME == Games.ENDURO:
                    new_obs, r, truncated, terminated, _ = out
                    # new_obs = self.reshape_obs(new_obs)
                    done = truncated or terminated
                else:
                    new_obs, r, done, _ = out  #type: ignore

                # save the experience
                if self.save_episodes:
                    actions.append(action)
                    rewards.append(r)
                    observations.append(obs)

                # update the loop
                obs = new_obs
                total_reward += r
                steps += 1

            total_steps.append(steps)
            total_rewards.append(total_reward)

            env.close()

            # save the episode to disk
            if self.save_episodes:

                observations = np.array(observations, dtype=np.uint8)
                actions = np.array(actions, dtype=np.float16)
                rewards = np.array(rewards, dtype=np.float16)

                np.savez_compressed(
                    save_file,  # type: ignore
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                )

        return total_steps, total_rewards

    # def reshape_obs(self, obs: np.ndarray):
    #     obs = obs[15:155, 10:, :]


class ResizeFrame(gym.ObservationWrapper):
    """
    Environment wrapper which alters the frame images.
    Removes the bottom section containing score and other info and 
    resizes the frame to 64x64
    """

    def __init__(self, env, imwidth=64, imheight=64):
        super().__init__(env)
        self.imwidth = imwidth
        self.imheight = imheight
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(self.imwidth, self.imheight, 3))

    def observation(self, frame: np.ndarray) -> np.ndarray:
        # cut the bottom section of the image containing the score
        frame = frame[15:155, 10:, :]
        # resize image with pillow to 64x64
        frame = np.array(
            Image.fromarray(frame).resize((self.imwidth, self.imheight)))
        return frame


class DreamChaserProcess(DreamGame, multiprocessing.Process):

    def __init__(
        self,
        agent: Optional[Dreamer] = None,
        controller: Optional[controller.C] = None,
        render: bool = True,
        save_episodes: bool = False,
        n_games: int = 1,
        result_queue: Optional[multiprocessing.Queue] = None,
        process_id: Optional[int] = None,
    ):
        self.agent = agent
        self.controller = controller
        self.n_games: int = n_games
        self.result_queue: Optional[multiprocessing.Queue] = result_queue
        self.process_id: Optional[int] = process_id
        DreamGame.__init__(self,
                           agent=agent,
                           render=render,
                           save_episodes=save_episodes)
        multiprocessing.Process.__init__(self)

    def run(self):
        # print(f"Started process {self.pid} to play {self.n_games} games")
        # log_every = self.n_games // 5
        # total_steps = []
        # total_reward = []

        if self.agent is None:
            self.agent = Dreamer(strategy=Strategy.MODEL)
            self.agent.load_memory()
            self.agent.load_vae()
            if controller is not None:
                self.agent.load_controller(self.controller)

        start_time = timeit.default_timer()
        steps, rewards = DreamGame.play(self, num_episodes=self.n_games)

        final_time = timeit.default_timer() - start_time
        if self.result_queue is not None and self.process_id is not None:
            print(f"[{self.process_id+1} / {cfg.POPULATION}] "
                  f"Final mean reward: {np.mean(rewards)}")
            result = {}
            result[self.process_id] = (np.mean(rewards), final_time)
            self.result_queue.put(result)
        return rewards


def frames_to_video(frames: Union[torch.Tensor, np.ndarray],
                    filename: Union[str, Path]):

    if isinstance(frames, torch.Tensor):
        frames = (frames.detach().cpu().permute(0, 2, 3, 1) *
                  255).long().numpy().astype(np.uint8)

    out = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"DIVX"), 20,
                          (64, 64))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  #type: ignore
        out.write(frame)
    out.release()


if __name__ == "__main__":
    gym.logger.setLevel(gym.logger.ERROR)

    # game_type = Games.ENDURO
    dreamer = Dreamer(strategy=Strategy.RANDOM)
    game = DreamGame(agent=dreamer, render=True, save_episodes=False)
    game.play()
