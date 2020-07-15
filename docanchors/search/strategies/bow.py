"""Strategies for documents in bag-of-words representation.

This is as in Ribeiro et al. 2018 for TextAnchor.
"""

import numpy as np

from .strategy import Strategy


class Add(Strategy):
    """Add one or more words.

    With *num = 1* (the default), this is the original Anchor search strategy proposed in [1].

    Parameters
    ----------
    num: int
        Number of words to add.

    References
    ----------
    [1] Ribeiro et al., 2018: Anchors: High-Precision Model-Agnostic Explanations.
    """

    def __init__(self, num: int = 1):
        super(Add, self).__init__()
        self.num = num

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        positions = np.random.choice(np.flatnonzero(~candidate), size=self.num)
        child = candidate.copy()
        child[positions] = True
        return child


class Flip(Strategy):
    """Flip one or more words.

    Parameters
    ----------
    num: int
        Number of words to flip.
    """

    def __init__(self, num: int = 1):
        super(Flip, self).__init__()
        self.num = num

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        position = np.random.choice(range(len(candidate)), size=self.num)
        child = candidate.copy()
        child[position] = ~child[position]
        return child


class Remove(Strategy):
    """Remove one or more words.

    Parameters
    ----------
    num: int
        Number of words to remove.
    """

    def __init__(self, num: int = 1):
        super(Remove, self).__init__()
        self.num = num

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        position = np.random.choice(np.flatnonzero(candidate), size=self.num)
        child = candidate.copy()
        child[position] = False
        return child
