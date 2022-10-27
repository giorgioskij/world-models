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

    def sample():  #type: ignore
        return random.sample(list(ChaserAction.__members__.values()),
                             1)[0].value


class DodgeballAction(Enum):
    LEFT = 1
    RIGHT = 7
    UP = 5
    DOWN = 3
    NOOP = 4
    THROW = 9

    def sample():  #type: ignore
        return random.sample(list(DodgeballAction.__members__.values()),
                             1)[0].value


class Games(Enum):
    CHASER = ("procgen:procgen-chaser-v0", len(ChaserAction))
    DODGEBALL = ("procgen:procgen-dodgeball-v0", len(DodgeballAction))

    def __init__(self, longname: str, num_actions: int):
        self.longname = longname
        self.num_actions = num_actions

    def sampleAction(self):
        if self == Games.CHASER:
            return ChaserAction.sample()
        elif self == Games.DODGEBALL:
            return DodgeballAction.sample()
        else:
            raise NotImplementedError