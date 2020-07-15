import numpy as np


class Replacement:

    def __call__(self, original: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UnknownValue(Replacement):
    # TODO: Allow to set custom UNK-token?

    def __call__(self, original: np.ndarray) -> np.ndarray:
        return np.zeros_like(original)

# TODO: More advanced replacement methods
# TODO: Documentation & literature
