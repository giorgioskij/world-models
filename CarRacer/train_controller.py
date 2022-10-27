"""
Instead of the evolution strategies proposed in the paper, let's use a policy
gradient strategy: it should work well with continuous action spaces
(contrarily to value-based policies, like DDQ-learning), and hopefully it's more
efficient than CMA-ES.
"""

from collections import deque
from typing import List
import game
from dreamer import Dreamer
import torch
import numpy as np
import controller
import config


def reinforce(agent: Dreamer, optimizer, n_train_episodes,
              gamma) -> List[float]:
    """
        Implementation of REINFORCE, a policy gradient algorithm to train the
        controller.

        Args:
            agent (Dreamer): The agent to train. Has to implement a method 
                find_move(obs, a) which returns the best move given the current
                observation.
            optimizer (torch.optim.Optimizer): The optimizer to use for training        
            n_train_episodes (int): Number of episodes to train on
            gamma (float): Discount factor

        Returns:
            List[float]: The average reward over the training episodes

    """
    scores_deque = deque(maxlen=100)
    scores = []

    for episode in range(n_train_episodes):

        render = episode % 5 == 0
        # play a rollout with the current policy
        rewards, log_probs = game.play_rollout(strategy=agent.find_move,
                                               render=render)

        # calculate discounted rewards
        rewards = torch.tensor(rewards)
        log_probs = torch.stack(log_probs)

        discounted_rewards_to_go = torch.empty_like(rewards).to(
            log_probs.device)

        # calculate discounted rewards to go
        for i in range(len(rewards)):
            n_to_go = len(rewards) - i
            gammas = torch.full((n_to_go,), gamma)
            discounted_gammas = torch.pow(gammas, torch.arange(n_to_go))
            discounted_reward = torch.sum(rewards[i:] * discounted_gammas)
            discounted_rewards_to_go[i] = discounted_reward

        discounted_probs = log_probs * discounted_rewards_to_go.expand_as(
            log_probs.t()).t()

        # calculate loss
        loss = -torch.sum(discounted_probs)

        # update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 1 == 0:
            print(f"Episode {episode+1}\tScore {sum(rewards):.2f}")

    return scores


c = controller.C(**config.CONTROLLER)
agent = Dreamer(c=c, train_controller=True)
optimizer = torch.optim.Adam(params=c.parameters(), lr=1e-2)
scores = reinforce(agent, optimizer, 1000, 0.99)
