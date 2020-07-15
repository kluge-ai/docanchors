import numpy as np

from .strategy import Strategy


class Seed(Strategy):
    """

    Parameters
    ----------
    num_seeds : int
        Number of seeds to plant.
    size : int
        Size of each seed.
    """

    def __init__(self, num_seeds: int = 1, size: int = 1):
        super(Seed, self).__init__()
        self.num_seeds = num_seeds
        self.size = size

    def __call__(self, candidate: np.ndarray):
        new_candidate = candidate.copy()
        possible_indices = np.nonzero(~candidate)[0]
        if len(possible_indices):
            seed_indices = self._random.choice(possible_indices, size=self.num_seeds)
        else:
            return new_candidate

        for idx in seed_indices:
            new_candidate[idx:idx + self.size] = True
        return new_candidate
