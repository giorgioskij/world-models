"""
A collection of enums to describe the games implemented and the available 
action for each of them.

When adding a new game, the first thing to do is to extend this file accordingly.
"""
from enum import Enum
import random


class ChaserAction(Enum):
    LEFT_DOWN = 0
    LEFT = 1
    LEFT_UP = 2
    DOWN = 3
    NONE = 4
    UP = 5
    RIGHT_DOWN = 6
    RIGHT = 7
    RIGHT_UP = 8

    @staticmethod
    def sample(): 
        return random.sample(list(ChaserAction.__members__.values()),
                             1)[0].value


class DodgeballAction(Enum):
    LEFT = 1
    RIGHT = 7
    UP = 5
    DOWN = 3
    NOOP = 4
    THROW = 9

    @staticmethod
    def sample():
        return random.sample(list(DodgeballAction.__members__.values()),
                             1)[0].value


class EnduroAction(Enum):
    NOOP = 0
    UP = 1
    RIGHT = 2
    LEFT = 3
    DOWN = 4

    @staticmethod
    def sample(): 
        return random.sample(list(EnduroAction.__members__.values()),
                             1)[0].value


class Games(Enum):
    CHASER = ("procgen:procgen-chaser-v0", len(ChaserAction))
    DODGEBALL = ("procgen:procgen-dodgeball-v0", len(DodgeballAction))
    ENDURO = ("ALE/Enduro-v5"), len(EnduroAction)
    

    def __init__(self, longname: str, num_actions: int):
        self.longname = longname
        self.num_actions = num_actions

    def sample_action(self):
        match self:
            case Games.CHASER: 
                return ChaserAction.sample()
            case Games.DODGEBALL:
                return DodgeballAction.sample()
            case Games.ENDURO:
                return EnduroAction.sample()
            case _:
                raise NotImplementedError