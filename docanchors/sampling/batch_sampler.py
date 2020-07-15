import time
import warnings
from multiprocessing import Queue
from queue import Full
from typing import Callable, Union
import logging

import numpy as np

from .masking import Mask, RandomUniformMask
from .replacement import Replacement, UnknownValue
from .sampler import Sampler
from ..util.util import get_final_padding_length, left_align


# TODO: Replace warnings with proper log messages
# TODO: Expose metrics for queues and sampling success rate
# TODO: Test if it works with Jupyter notebooks/find a way to run it

class BatchSampler(Sampler):
    # TODO: Documentation
    """

    Parameters
    ----------
    instance : np.ndarray
    get_predict_fn : Callable
    sample_queue : Queue
    batch_size : int
    mask: Mask
    replacement : Replacement
    restrict_to_known : bool
    target: None, int
    """

    def __init__(self,
                 instance: np.ndarray,
                 get_predict_fn: Callable[[], Callable[[np.ndarray], int]],
                 sample_queue: Queue,
                 batch_size: int = 256,
                 mask: Mask = RandomUniformMask(),
                 replacement: Replacement = UnknownValue(),
                 restrict_to_known: bool = True,
                 left_align: bool = False,
                 target: Union[None, int] = None,
                 max_samples: Union[None, int] = None):
        super(BatchSampler, self).__init__()

        self.instance = instance
        self.model_input_shape = instance.shape

        self.predict_fn = None
        self.get_predict_fn = get_predict_fn

        predict_fn = self.get_predict_fn()
        self.target = target or predict_fn(instance.reshape(1, -1))[0]
        del predict_fn

        self.single_queue = sample_queue

        self.model_batch_size = batch_size
        self.mask = mask
        self.replacement = replacement

        self.restrict_to_known = restrict_to_known
        self.final_padding_start = len(instance) - get_final_padding_length(instance)

        self.left_align = left_align
        self.max_samples = max_samples

    def run(self):
        self.predict_fn = self.get_predict_fn()

        batch_shape = (self.model_batch_size, *self.model_input_shape)
        batch_base = np.repeat(self.instance.reshape(1, -1), self.model_batch_size, axis=0)
        assert batch_base.shape == batch_shape

        previous = 0.0

        cnt = 0

        while True:
            if self.single_queue.full():
                self.logger.warn("Sample queue is full (overflow).")
                time.sleep(0.1)
                continue

            batch_mask = self.mask(batch_shape)
            if self.restrict_to_known:
                batch_mask[:, self.final_padding_start:] = True
            model_input = np.where(batch_mask, batch_base, self.replacement(batch_base))
            if self.left_align:
                model_input = left_align(model_input)
            model_output = self.predict_fn(model_input).ravel()
            is_target_sample = np.array(model_output == self.target)
            deviation = np.mean(is_target_sample) - 0.5

            # if deviation is positive, we have too many target samples -> increase perturbation rate
            # if deviation is negative, we have too few target samples -> decrease perturbation rate
            # TODO: Adjust pre-factor
            self.mask.perturbation_rate += 0.01 * deviation + 0.0001 * (deviation - previous)
            self.mask.perturbation_rate = max(0.10, min(self.mask.perturbation_rate, 0.90))
            previous = deviation

            if abs(deviation) > 0.25 and 0.10 < self.mask.perturbation_rate < 0.90:
                self.logger.debug(f"Warmup, deviation {deviation}")
                continue

            self.mask.learn(mask=batch_mask, is_target_sample=is_target_sample.astype(int))

            if self.single_queue.empty():
                self.logger.warn("Sample queue is empty (underflow).")

            if self.single_queue.full():
                continue

            # TODO: Investigate and if advantageous refactor to use sparse matrices
            for sample, label in zip(batch_mask, is_target_sample.astype(int)):
                try:
                    self.single_queue.put((sample, label), block=False)
                except Full:
                    break

            cnt += batch_mask.shape[0]

            if self.max_samples is not None and cnt > self.max_samples:
                break
