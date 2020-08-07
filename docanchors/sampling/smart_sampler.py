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
                 get_predict_fn: Callable[[], Callable[[np.ndarray], np.ndarray]],
                 sample_queue: Queue,
                 batch_size: int = 256,
                 target: Union[None, int] = None,
                 min_tokens: int = 1,
                 max_tokens: Union[None, int] = None,
                 replacement: Replacement = UnknownValue(),
                 restrict_to_known: bool = True,
                 left_align: bool = True,
                 max_samples: Union[None, int] = None):
        super(SmartSampler, self).__init__()
        self.instance = instance

        self.predict_fn = None
        self.get_predict_fn = get_predict_fn

        predict_fn = self.get_predict_fn()
        self.target = target or predict_fn(instance.reshape(1, -1))[0]
        del predict_fn

        self.single_queue = sample_queue

        self.model_batch_size = batch_size
        self.replacement = replacement

        self.restrict_to_known = restrict_to_known
        self.final_padding_start = len(instance) - get_final_padding_length(instance)

        self.left_align = left_align
        self.max_samples = max_samples

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens or len(self.instance)

        self._rng = np.random.default_rng()

    def run(self):
        self.predict_fn = self.get_predict_fn()

        batch_shape = (self.model_batch_size, len(self.instance))
        batch_base = np.repeat(self.instance.reshape(1, -1), self.model_batch_size, axis=0)
        assert batch_base.shape == batch_shape

        if self.restrict_to_known:
            size = self.final_padding_start
        else:
            size = len(self.instance)

        windows = []
        for window_size in range(self.min_tokens, self.max_tokens):
            window = np.zeros(shape=(size, len(self.instance)), dtype=bool)
            for i, row in enumerate(window):
                if i + window_size > size:
                    break
                row[i:i + window_size] = True
            windows.append(window)

        all_batches = np.concatenate(windows)

        self.logger.info(f"Generated {all_batches.shape[0]} samples.")

        if self.max_samples is not None:
            if all_batches.shape[0] > self.max_samples:
                all_batches = self._rng.choice(all_batches, size=self.max_samples,
                                               replace=False, axis=0)

        self._rng.shuffle(all_batches, axis=0)

        self.logger.info(f"Truncated to {all_batches.shape[0]} samples.")
        model_calls = int(all_batches.shape[0] / self.model_batch_size) + 1

        for batch_count in range(model_calls):
            self.logger.info(f"Batch {batch_count} of {model_calls}")
            batch_start = batch_count * self.model_batch_size
            batch_mask = all_batches[batch_start:batch_start + self.model_batch_size, :]

            if batch_mask.shape[0] == 0:
                break

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
                # print(label, sample[:5])
