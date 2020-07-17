from typing import Dict, List

import numpy as np

from ...util.configuration.component import Component


class Objective(Component):

    def __init__(self, pre_factor=1.0):
        super(Objective, self).__init__()
        self.pre_factor = pre_factor
        self.name = self.__class__.__name__

    def value(self, candidate: np.ndarray) -> float:
        raise NotImplementedError

    def log_value(self, candidate: np.ndarray) -> Dict[str, float]:
        _log_value = self(candidate)
        return {self.name: _log_value}

    def __call__(self, candidate: np.ndarray) -> float:
        return self.pre_factor * self.value(candidate)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.pre_factor *= other
            return self
        else:
            raise ValueError(f"Can only multiply Objectives with int or float, not {type(other)}.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Objective):
            return Join([self, other])
        else:
            raise ValueError(f"Can only add Objectives to Objectives, not {type(other)}.")

    def __radd__(self, other):
        return self.__add__(other)


class Join(Objective):

    def __init__(self, objectives: List[Objective]):
        super(Join, self).__init__()
        self.objectives = objectives

    def value(self, candidate: np.ndarray) -> float:
        return self.pre_factor * sum((objective(candidate)
                                      for objective in self.objectives))

    def log_value(self, candidate: np.ndarray) -> Dict[str, float]:
        value = {}
        for objective in self.objectives:
            value.update(objective.log_value(candidate))
        return value
