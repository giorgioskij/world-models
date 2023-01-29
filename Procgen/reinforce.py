"""
An implementation of the very popular REINFORCE algorithm.
I used it in various occasions for reference, and I tried to train the
controller on the game Enduro with this algorithm.
"""
from game import ResizeFrame
from implemented_games import Games
from typing import *
import torch
from torch import Tensor
import config as cfg
import gym
from controller import C
from pathlib import Path
from memory import M
from vae import V
from torch.nn import functional as F


def reinforce(
    controller: C,
    optimizer,
    n_train_episodes: int,
    gamma,
    render_every: int = -1,
    log_every: int = 1,
    test: bool = False,
    m_path: Path = cfg.CKP_DIR / "memory.pt",
    v_path: Path = cfg.CKP_DIR / "vae.pt",
) -> Tuple[List[float], List[float]]:
    """
    NOTE: the lines regarding the memory are currently commented out, because
    I tried to train the controller passing as input ONLY the z_vector.
    Expectedly, it did not improve anything. 
    """

    losses: List[float] = []
    total_rewards: List[float] = []

    # load vae and memory
    v_model: V = V(**cfg.VAE).to(cfg.DEVICE)
    # m_model: M = M(**cfg.LSTM).to(cfg.DEVICE)
    v_model.load_state_dict(torch.load(v_path))
    # m_model.load_state_dict(torch.load(m_path))
    v_model.eval()
    # m_model.eval()

    controller.to(cfg.DEVICE)
    controller.train()

    env = gym.make(cfg.GAME.longname, render_mode="rgb_array")
    if cfg.GAME == Games.ENDURO:
        env = ResizeFrame(env)

    for episode in range(n_train_episodes):

        render = (episode > 0 and render_every > 0 and
                  episode % render_every == 0)
        if render:
            env.close()
            env = gym.make(cfg.GAME.longname, render_mode="human")

        # let's play a game the old fashioned way to make it more clear
        obs = env.reset()
        if cfg.GAME == Games.ENDURO:
            obs = obs[0]

        done: bool = False
        steps: int = 0
        log_probs: List[Tensor] = []
        rewards: List[float] = []
        while not done and steps < cfg.MAX_STEPS:

            # every tot. steps, re-initialize agent memory
            # if steps % 100 == 0 or m_model.hidden is None:
            #     m_model.init_hidden(1)

            # convert observation
            obs = (torch.from_numpy(obs).float() / 255).reshape(
                1, cfg.IMHEIGHT, cfg.IMWIDTH, 3).permute(0, 3, 1,
                                                         2).to(cfg.DEVICE)

            with torch.no_grad():
                # get latent state
                z, _, _, = v_model.encode(obs)

                # concatenate z and lstm hidden state
                # z_h = torch.cat((z.squeeze(), m_model.hidden[0].squeeze()),
                #                 dim=-1)

            # choose action with the controller
            # logits = controller(z_h)
            logits = controller(z)

            distribution = torch.distributions.Categorical(logits=logits)
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)

            # concatenate action and latent vector to update memory
            # action_one_hot = F.one_hot(action_tensor,
            #                            num_classes=cfg.GAME.num_actions)
            # z_a = torch.cat((z.squeeze(), action_one_hot), dim=-1)

            # update memory
            # with torch.no_grad():
            #     _ = m_model(z_a.unsqueeze(0), use_hidden=True)

            # perform the action and get the reward and new obs
            action = int(action_tensor.item())
            out = env.step(action)  # type: ignore
            if cfg.GAME == Games.ENDURO:
                new_obs, r, truncated, terminated, _ = out
                done = truncated or terminated
            else:
                new_obs, r, done, _ = out  #type: ignore

            # update the loop
            obs = new_obs
            steps += 1
            rewards.append(r)
            log_probs.append(log_prob)

        # end of the game played
        env.close()

        # calculate discounted rewards
        rewards_tensor = torch.tensor(rewards)

        # add decreasing reward over time
        rewards_tensor -= torch.arange(len(rewards_tensor)) * 0.1

        log_probs_tensor = torch.stack(log_probs)
        # discounted_rewards = torch.empty_like(rewards_tensor).to(cfg.DEVICE)

        # calculate discounted rewards
        returns = []
        R = 0
        for r in rewards_tensor.flip(dims=(0,)):
            R = r + gamma * R
            returns.append(R)
        returns = torch.tensor(returns[::-1]).to(cfg.DEVICE)

        discounted_probs = log_probs_tensor * returns

        # calculate loss
        loss = -torch.sum(discounted_probs)

        # update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update loop
        losses.append(loss.item())
        total_rewards.append(rewards_tensor.sum().item())

        if (episode + 1) % log_every == 0:
            print(f"Episode {episode+1}\tScore {sum(rewards_tensor):.2f}")

    return losses, total_rewards


if __name__ == "__main__":
    c = C(**cfg.CONTROLLER)
    optimizer = torch.optim.Adam(params=c.parameters(), lr=1e-2)
    try:
        losses, total_rewards = reinforce(
            c,
            optimizer,
            n_train_episodes=5000,
            gamma=0.9,
            render_every=-1,
            log_every=1,
        )
    except KeyboardInterrupt as k:
        print("Stopping training...")
        torch.save(c.state_dict(), cfg.CKP_DIR / "reinforce-controller.pt")
        raise k

    # scores = reinforce(agent, optimizer, 1000, 0.99)
