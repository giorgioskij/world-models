""" run processes in parallel to collect random episodes"""

from implemented_games import Games
from game import DreamChaserProcess
from agent import Dreamer, Strategy
import os
import signal
import gym


def terminate_processes(process_list):
    for x in process_list:
        x.terminate()
        print(f"Killed {x.pid}")


if __name__ == "__main__":
    gym.logger.setLevel(gym.logger.ERROR)

    dreamer = Dreamer(strategy=Strategy.RANDOM,)

    print(f"Main PID: {os.getpid()}")
    n_processes = 1
    total_games = 10
    games_per_process = total_games // n_processes

    processes = []
    for i in range(n_processes):
        p = DreamChaserProcess(agent=dreamer,
                               render=False,
                               save_episodes=True,
                               n_games=games_per_process)
        p.start()
        processes.append(p)

    signal.signal(signal.SIGINT, lambda x, y: terminate_processes(processes))

    for p in processes:
        p.join()
        print(f"Collected process {p.pid}")
