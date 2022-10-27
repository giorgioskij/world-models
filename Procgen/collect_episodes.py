import config as cfg
from implemented_games import Games
from game import DreamChaserProcess
from agent import Dreamer, Strategy
from torch import multiprocessing as multiprocessing
import os
import signal
import gym


def terminate_processes(processes):
    for p in processes:
        p.terminate()
        print(f"Killed {p.pid}")


if __name__ == "__main__":
    gym.logger.setLevel(gym.logger.ERROR)

    dreamer = Dreamer(
        game=Games.DODGEBALL,
        strategy=Strategy.RANDOM,
    )

    print(f"Main PID: {os.getpid()}")
    n_processes = 10
    total_games = 10000
    games_per_process = total_games // n_processes

    processes = []
    for i in range(n_processes):
        p = DreamChaserProcess(game=Games.DODGEBALL,
                               agent=dreamer,
                               render=False,
                               save_episodes=True,
                               n_games=games_per_process)
        p.start()
        processes.append(p)

    signal.signal(signal.SIGINT, lambda x, y: terminate_processes(processes))

    for p in processes:
        p.join()
        print(f"Collected process {p.pid}")
