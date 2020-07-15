import numpy as np

from .strategy import Strategy


class UniformRandom(Strategy):

    def __init__(self, rate: float = 0.5):
        super(UniformRandom, self).__init__()
        self.rate = rate

    def __call__(self, candidate: np.ndarray):
        return self._random.random(size=len(candidate)) < self.rate
