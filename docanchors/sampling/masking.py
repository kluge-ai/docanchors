from typing import Tuple, Any, Union

import numpy as np

EPSILON = 1e-12


class Mask:

    def __init__(self, perturbation_rate: float = 0.5):
        self.perturbation_rate = perturbation_rate

    def __call__(self, batch_shape: Tuple[int, Any]) -> np.ndarray:
        raise NotImplementedError

    def learn(self, mask: np.ndarray, is_target_sample: np.ndarray):
        pass

    def reset(self):
        pass


class RandomUniformMask(Mask):

    def __call__(self, batch_shape: Tuple[int, Any]) -> np.ndarray:
        return np.random.random_sample(batch_shape) >= self.perturbation_rate


class RandomChunkMask(Mask):

    def __init__(self, perturbation_rate: float = 0.5, chunk_size: int = 2):
        super(RandomChunkMask, self).__init__(perturbation_rate=perturbation_rate)
        self.chunk_size = chunk_size

    def __call__(self, batch_shape: Tuple[int, Any]) -> np.ndarray:
        base_mask = np.random.random_sample(batch_shape) >= self.perturbation_rate
        chunked_mask = np.repeat(base_mask, self.chunk_size, axis=1)
        offset = np.random.randint(0, self.chunk_size)
        return chunked_mask[:, offset:batch_shape[1] + offset]


class RandomNonUniformMask(Mask):

    def __init__(self, perturbation_rate: float = 0.5,
                 probabilities: Union[np.ndarray, None] = None,
                 size: int = 1):
        super(RandomNonUniformMask, self).__init__(perturbation_rate=perturbation_rate)
        self.probabilities = probabilities
        self.size = size

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, value: np.ndarray):
        self._probabilities = value / np.sum(value)

    def __call__(self, batch_shape: Tuple[int, Any]) -> np.ndarray:
        if len(batch_shape) == 2:
            mask = np.zeros(shape=batch_shape, dtype=bool)
            if batch_shape[1] >= len(self.probabilities):
                size = max(2, int((1 - self.perturbation_rate) * batch_shape[1] / self.size))
                tokens = list(range(len(self.probabilities)))
                for row in mask:
                    indices = np.random.choice(tokens, size=size, p=self.probabilities)
                    indices = np.concatenate([indices + i for i in range(-int(size / 2), int(size / 2))])
                    indices = indices[(0 <= indices) & (indices < len(row))]
                    row[indices] = True
            else:
                raise ValueError("More probabilities than batch shape")
            return mask
        else:
            raise ValueError("Non-Uniform mask only works for one-dimensional samples")
