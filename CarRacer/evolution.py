"""
    Train the controller using CMA-ES.
    To make it a bit faster, the main function will spawn a number of child 
    processes, and divide the population between them.
    The main process will not perform any evaluation, therefore it is crucial 
    that CUDA is NEVER initialized in the main process to avoid a huge waste of 
    memory for its initialization.

    The code is taken and adapted from:
    https://github.com/dylandjian/retro-contest-sonic/blob/master/train_controller.py
"""

from typing import Optional
from controller import C
import game
import config as cfg
import numpy as np
import cma
import torch
import math

from torch.multiprocessing import Queue
import torch.multiprocessing as multiprocessing

DEV = torch.device("cpu")


def init_controller(c, solution):
    """ Change weights of the controller to the ones proposed by CMA
    """
    new_w1 = torch.tensor(solution, dtype=torch.double, device=DEV)
    params = c.state_dict()
    params["fc.weight"].data.copy_(new_w1.view(-1, cfg.CONTROLLER["input_dim"]))


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def rankmin(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def create_results(result_queue, fitlist):
    """ Empty the res queue to adapt the fitlist, and get timers"""
    times = []
    while not result_queue.empty():
        result = result_queue.get()
        keys = list(result.keys())
        result = list(result.values())
        fitlist[keys[0]] = result[0][0]
        times.append(result[0][1])
    return times


def test_controller(c: C):
    # multiprocessing.set_start_method('spawn')
    q = Queue()
    new_game = game.VAECGame(process_id=0,
                             controller=c,
                             result_queue=q,
                             render_mode="rgb-array")
    new_game.rungame()
    result = q.get()
    results = list(result.values())
    score = results[0][0]
    return score


def train_controller(render_sometimes: bool = True):
    """ Train the controller using CMA-ES, with parallel batch evaluation.
        The code for this function is adapted from:
        https://github.com/dylandjian/retro-contest-sonic
    """

    multiprocessing.set_start_method('spawn')
    n_parallel = cfg.PARALLEL
    popsize = cfg.POPULATION
    best_c = C(**cfg.CONTROLLER)

    num_controller_params = sum(p.numel() for p in best_c.parameters())
    solver = CMAES(num_controller_params, sigma_init=4, popsize=popsize)
    result_queue = Queue()

    number_generations = 1
    current_best = -math.inf
    current_ctrl_version = 1

    for _ in range(1000):
        solutions = solver.ask()
        fitlist = np.zeros(popsize)
        eval_left = 0

        while eval_left < popsize:
            if eval_left == 0 and render_sometimes:
                MODE = "human"
            else:
                MODE = "rgb-array"
            jobs = []
            todo = n_parallel if eval_left + n_parallel <= popsize else (
                eval_left + n_parallel) % popsize

            # spawn the processes
            print("[CONTROLLER] Starting new batch")
            for job in range(todo):
                process_id = eval_left + job

                # assign new weights to the controller, given by the CMA
                c = C(**cfg.CONTROLLER)
                init_controller(c, solutions[process_id])

                # start the evaluation
                new_game = game.VAECGame(process_id=process_id,
                                         controller=c,
                                         result_queue=result_queue,
                                         render_mode=MODE)
                # new_game.run()
                new_game.start()

                jobs.append(new_game)

            # collect the processes
            for p in jobs:
                p.join()

            eval_left += todo
            print("[CONTROLLER] done with batch")

        print("End of population")
        # collect the results
        times = create_results(result_queue, fitlist)

        # metrics
        current_score = np.max(fitlist)
        average_score = np.mean(fitlist)

        # update solver
        max_idx = np.argmax(fitlist)
        fitlist = rankmin(fitlist)
        print("Feeding stuff to the solver")
        solver.tell(fitlist)
        new_results = solver.result

        ## Display
        print("[CONTROLLER] Total duration for generation: %.3f seconds, average duration:"
            " %.3f seconds per process, %.3f seconds per run" % ((np.sum(times), \
                    np.mean(times), np.mean(times) / cfg.REPEAT_ROLLOUT)))
        print("[CONTROLLER] Creating generation: {} ...".format(
            number_generations + 1))
        print("[CONTROLLER] Current best score: {}, new run best score: {}".
              format(current_best, current_score))
        print(
            "[CONTROLLER] Best score ever: {}, current number of improvements: {}"
            .format(current_best, current_ctrl_version))
        print("[CONTROLLER] Average score on all of the processes: {}\n".format(
            average_score))

        # save the best controller
        if current_score > current_best:
            init_controller(best_c, solutions[max_idx])
            state = {
                "version": current_ctrl_version,
                "score": current_score,
                "generation": number_generations,
            }
            save_checkpoint(best_c, state)
            current_ctrl_version += 1
            current_best = current_score


def save_checkpoint(c, state):

    filename = cfg.CKP_DIR / f"{state['version']}-controller.pth.tar"
    state["model"] = c.state_dict()
    torch.save(state, filename)


def load_checkpoint(path):
    cp = torch.load(path)
    c = C(**cfg.CONTROLLER)
    c.load_state_dict(cp["model"])
    return c, cp


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES:
    """ Wrapper for the CMA-ES algorithm. The code for this class was taken from
        https://github.com/dylandjian/retro-contest-sonic
    """

    def __init__(self,
                 num_params,
                 sigma_init=0.5,
                 popsize=255,
                 weight_decay=0.0):
        """
        Sigma: initial standard deviation
        Popsize: size of the population
        Num_params: number of parameters in the candidate solution
        Weight_decay: modify the reward according the a weighted mean of the parameters
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None
        self.es = cma.CMAEvolutionStrategy(self.num_params * [0],
                                           self.sigma_init,
                                           {'popsize': self.popsize})

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        """ Ask the CMA-ES to sample new candidate solutions """

        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        """ Give the reward result to the CMA-ES to adapt mu and sigma """

        reward_table = -np.array(reward_table_result)

        ## Compute reward decay if needed
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        self.es.tell(self.solutions, (reward_table).tolist())

    def current_param(self):
        """ Mean solution, better with noise """

        return self.es.result[5]

    def best_param(self):
        """ Best evaluated solution, useful for saving the best agent """

        return self.es.result[0]

    def result(self):
        """ Best params so far, along with historically best reward, curr reward, sigma """

        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])


if __name__ == "__main__":
    # train_controller()
    c, cp = load_checkpoint(cfg.CKP_DIR / "30-controller.pth.tar")
    scores = []
    for _ in range(100):
        score = test_controller(c)
        scores.append(score)
