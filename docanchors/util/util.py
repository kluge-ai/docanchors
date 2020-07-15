from typing import Dict, Any

import numpy as np


def get_final_padding_length(instance: np.ndarray,
                             unknown_token: int = 0):
    unknown_values = instance == unknown_token
    return np.argmax(np.diff(unknown_values)[::-1]) + 1


def left_align(samples: np.ndarray):
    new_samples = np.zeros_like(samples)
    for i, row in enumerate(new_samples):
        sample = samples[i]
        non_zero = sample != 0
        row[:np.sum(non_zero)] = sample[non_zero]
    return new_samples


def get_instance_attributes(instance: object) -> Dict[str, Any]:
    return {attribute: value for attribute, value in instance.__dict__.items()
            if not attribute[0] == "_" and not str(hex(id(value))) in str(value)}

# TODO: Add queue generator
