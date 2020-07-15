from typing import List

import numpy as np

from .strategy import Strategy


class DivideAndEvolve(Strategy):

    def __init__(self, strategies: List[Strategy], weights: List[float], parts: int):
        super(DivideAndEvolve, self).__init__()
        self.strategies = strategies
        self.weights = np.array(weights) / np.sum(weights)
        if len(self.strategies) != len(self.weights):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of strategies ({len(self.strategies)})")
        self.parts = parts

    def __call__(self, candidate: np.ndarray):
        split_indices = np.sort(np.random.choice(range(len(candidate)), self.parts - 1))
        cut_idx = [0] + list(split_indices) + [len(candidate)]
        parts = [candidate[cut_idx[i]:cut_idx[i + 1]] for i in range(self.parts)]
        strategies = self._random.choice(self.strategies, size=self.parts, replace=True, p=self.weights)
        return np.concatenate([strategy(part) for strategy, part in zip(strategies, parts)
                               if len(part) > 0])
