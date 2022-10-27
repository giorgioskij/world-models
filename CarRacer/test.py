from dreamer import Dreamer
from game import play_rollout

d = Dreamer()

play_rollout(d.find_move, render=True)