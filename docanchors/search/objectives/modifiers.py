"""Objectives that modify other objectives."""
from typing import Dict

import numpy as np

from docanchors.search.objectives.objective import Objective

EPSILON = 1e-6


class Bound(Objective):
    """Bound the value of any `Objective`.

    Parameters
    ----------
    objective
    lower_bound
    upper_bound
    """

    def __init__(self, objective: Objective, lower_bound: float, upper_bound: float):
        super(Bound, self).__init__()
        self._objective = objective
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self._name = objective.name

    def value(self, candidate: np.ndarray) -> float:
        _value = self._objective(candidate)
        return min(max(_value, self.lower_bound), self.upper_bound)

    def log_value(self, candidate: np.ndarray) -> Dict[str, float]:
        return self._objective.log_value(candidate)


class Invert(Objective):
    """Inverse of any `Objective`.

    Parameters
    ----------
    objective : Objective
        The objective
    epsilon : float
        Constant to prevent ZeroDivision errors

    Examples
    --------

    >>> from docanchors.search.objectives import Conciseness
    >>> candidate = np.array([1, 1, 0, 0], dtype=bool)
    >>> conciseness = Conciseness()
    >>> conciseness(candidate)
    0.5
    >>> inverted_conciseness = Invert(conciseness)
    >>> inverted_conciseness(candidate)
    1.999996000008

    """

    def __init__(self, objective: Objective, epsilon: float = EPSILON):
        super(Invert, self).__init__()
        self._objective = objective
        self._epsilon = epsilon
        self._name = objective.name

    def value(self, candidate: np.ndarray) -> float:
        return 1 / (self._objective(candidate) + self._epsilon)

    def log_value(self, candidate: np.ndarray) -> Dict[str, float]:
        return self._objective.log_value(candidate)
