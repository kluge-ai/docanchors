"""Strategies for documents represented as arrays of tokens.

As presented in [1].

References
----------
[1] Kluge and Eckhardt, 2020: Explaining Suspected Phishing Attempts with Document Anchors
"""
import numpy as np

from .strategy import Strategy


class Grow(Strategy):
    """Grows highlights by adding an element after the last element of every highlight.

    Note that highlights at the end of the candidate are not affected.
    """

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        transitions = np.diff(candidate, n=1)

        transitions = np.concatenate(([False], transitions))

        return np.where(transitions,
                        np.ones_like(candidate, dtype=bool),
                        candidate)


class Shrink(Strategy):
    """Shrinks highlights by removing the last element of every highlight."""

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        transitions = np.diff(candidate, n=1)

        if candidate[-1]:
            transitions = np.append(transitions, True)
        else:
            transitions = np.append(transitions, False)

        return np.where(transitions,
                        np.zeros_like(candidate, dtype=bool),
                        candidate)


class Shift(Strategy):
    """Shift all highlights one element to the left or one element to the right.

    Direction is chosen randomly.
    """

    def __init__(self):
        super(Shift, self).__init__()
        self._shift_left = ShiftLeft()
        self._shift_right = ShiftRight()

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        if self._random.random() < 0.5:
            return self._shift_left(candidate)
        else:
            return self._shift_right(candidate)


class ShiftLeft(Strategy):

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        return np.append(candidate[1:], False)


class ShiftRight(Strategy):

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        return np.concatenate(([False], candidate[:-1]))


class Pass(Strategy):
    """Leave candidate unaltered."""

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        return candidate.copy()


class Erase(Strategy):
    """Remove all highlights from candidate."""

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        return np.zeros_like(candidate, dtype=bool)
