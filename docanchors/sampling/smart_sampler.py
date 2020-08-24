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


class SmartSampler(Sampler):
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
                 get_predict_fn: Callable[[float], Callable[[np.ndarray], np.ndarray]],
                 sample_queue: Queue,
                 batch_size: int = 256,
                 target: Union[None, int] = None,
                 min_radius: int = 0,
                 max_radius: Union[None, int] = None,
                 replacement: Replacement = UnknownValue(),
                 restrict_to_known: bool = True,
                 left_align: bool = True,
                 max_samples: Union[None, int] = None,
                 threshold: float = 0.8):
        super(SmartSampler, self).__init__()
        self.instance = instance

        self.predict_fn = None
        self.get_predict_fn = get_predict_fn

        self.threshold = threshold
        predict_fn = self.get_predict_fn(self.threshold)
        self.target = target or predict_fn(instance.reshape(1, -1))[0]
        del predict_fn

        self.single_queue = sample_queue

        self.model_batch_size = batch_size
        self.replacement = replacement

        self.restrict_to_known = restrict_to_known
        self.final_padding_start = len(instance) - get_final_padding_length(instance)

        self.left_align = left_align
        self.max_samples = max_samples

        self.min_radius = min_radius
        self.max_radius = max_radius or int(len(self.instance) / 2) - 1

        self._rng = np.random.default_rng()

    def run(self):
        self.predict_fn = self.get_predict_fn(self.threshold)

        if self.restrict_to_known:
            size = self.final_padding_start
        else:
            size = len(self.instance)

        all_batches = self.create_batches(size)
        self.logger.info(f"Generated {all_batches.shape[0]} samples.")

        batches = self.reduce_batches(all_batches)
        self.logger.info(f"Reduced to {batches.shape[0]} samples.")

        num_model_calls, remainder = divmod(batches.shape[0], self.model_batch_size)
        if remainder:
            num_model_calls += 1

        batch_shape = (self.model_batch_size, len(self.instance))
        batch_base = np.repeat(self.instance.reshape(1, -1), self.model_batch_size, axis=0)
        assert batch_base.shape == batch_shape

        for batch_count in range(num_model_calls):
            self.logger.info(f"Batch {batch_count + 1} of {num_model_calls}")
            batch_start = batch_count * self.model_batch_size
            batch_mask = batches[batch_start:batch_start + self.model_batch_size, :]

            if batch_mask.shape[0] < batch_base.shape[0]:
                batch_base = batch_base[:batch_mask.shape[0], :]

            model_input = np.where(batch_mask, batch_base, self.replacement(batch_base))
            if self.left_align:
                model_input = left_align(model_input)
            model_output = self.predict_fn(model_input).ravel()
            is_target_sample = np.array(model_output == self.target)

            self.logger.info(f"Target sample rate: {np.mean(is_target_sample):0.4f}")

            for sample, label in zip(batch_mask, is_target_sample.astype(int)):
                self.single_queue.put((sample, label), block=True)

    def reduce_batches(self, batches: np.ndarray) -> np.ndarray:
        if self.max_samples is not None:
            if batches.shape[0] > self.max_samples:
                batches = self._rng.choice(batches, size=self.max_samples,
                                           replace=False, axis=0)
        self._rng.shuffle(batches, axis=0)
        return batches

    def create_batches(self, size: int) -> np.ndarray:
        windows = []
        for radius in range(self.min_radius, self.max_radius):
            window_size = 2 * radius + 1
            window = np.zeros(shape=(size + window_size, len(self.instance)), dtype=bool)
            for i, row in enumerate(window):
                row[i - radius:i + radius] = True

            windows.append(window)
        all_batches = np.concatenate(windows)
        return all_batches
