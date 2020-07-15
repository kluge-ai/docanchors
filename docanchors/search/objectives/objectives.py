from multiprocessing import Queue
from typing import Union

import numpy as np

from .objective import Objective


EPSILON = 1e-6


class Zero(Objective):
    """Zero objective.

    Takes on 0.0 regardless of the candidate.
    """

    def value(self, candidate: np.ndarray) -> float:
        return 0.0


class Constant(Objective):
    """Constant objective.

    Takes on the specified value regardless of the candidate.
    """

    def __init__(self, constant_value: Union[float, int], **kwargs):
        super(Constant, self).__init__(**kwargs)
        self.constant_value = float(constant_value)

    def value(self, candidate: np.ndarray) -> float:
        return self.constant_value


class Conciseness(Objective):
    """Measures how sparse the candidate is.

    As introduced in [1].

    References
    __________
    [1] Lei at al., 2016: Rationalizing Neural Predictions (doi: 10.18653/v1/D16-1011)
    """

    def value(self, candidate: np.ndarray) -> float:
        return float(np.mean(candidate))


class Coherence(Objective):
    """Measures how connected the candidate is.

    As introduced in [1].

    References
    __________
    [1] Lei at al., 2016: Rationalizing Neural Predictions (doi: 10.18653/v1/D16-1011)
    """

    def value(self, candidate: np.ndarray) -> float:
        return float(np.sum(np.diff(candidate)))


class AbsoluteCover(Objective):
    """Measures how many non-zero tokens are in the candidate.

    As introduced in [1].

    Parameters
    ----------
    target : int
        Target coverage. If set to *0* (the default), shorter candidates are preferred.

    References
    ----------
    [1] Kluge and Eckhardt, 2020: Explaining Suspected Phishing Attempts with Document Anchors
    """

    def __init__(self, target: int = 0, **kwargs):
        super(AbsoluteCover, self).__init__(**kwargs)
        self.target = target

    def value(self, candidate: np.ndarray) -> float:
        return (np.sum(candidate) - self.target) ** 2


class RelativeCover(Objective):
    """Measures the fraction of non-zero tokens in the candidate.

    As introduced in [1].

    Parameters
    ----------
    target : float
        Target relative coverage. If set to *0.0* (the default), shorter candidates are preferred.

    References
    ----------
    [1] Kluge and Eckhardt, 2020: Explaining Suspected Phishing Attempts with Document Anchors
    """

    def __init__(self, target: float = 0.0, **kwargs):
        super(RelativeCover, self).__init__(**kwargs)
        self.target = target

    def value(self, candidate: np.ndarray) -> float:
        return (np.mean(candidate) - self.target) ** 2
