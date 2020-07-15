"""Some inspiration from https://github.com/Naereen/Kullback-Leibler-divergences-and-kl-UCB-indexes"""
from typing import List, Callable, Tuple
import numpy as np
import warnings
import logging

try:
    import pyximport
except ModuleNotFoundError:
    warnings.warn("Did not find Cython, falling back to slower Python implementation.")
    from .impl_python import compute_bound as _compute_bound
else:
    pyximport.install(language_level=3)
    from .impl_cython import compute_bound as _compute_bound

logger = logging.getLogger("KL-LUCB")

EPSILON = 1e-15
MAX_STEPS = 1000


# TODO: Complete documentation

def compute_bound(num_samples: int, observed_mean: float, divergence_bound: float,
                  previous_target_bound: float, previous_other_bound: float,
                  goal: Callable[[float, float], float], precision: float = 1e-3) -> float:
    """

    Parameters
    ----------
    num_samples
    observed_mean
    divergence_bound
    previous_target_bound
    previous_other_bound
    goal
    precision

    Returns
    -------

    """
    value = goal(previous_other_bound, observed_mean)
    target_bound = previous_target_bound

    return _compute_bound(num_samples, observed_mean, divergence_bound,
                          value, target_bound, precision)


def find_best_n(candidates: List[np.ndarray],
                n: int, min_delta: float,
                evaluate: Callable[[np.ndarray], int]) -> Tuple[List[np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    candidates : List of np.ndarrays
        The candidates to evaluate.
    n : int
        Number of candidates to select.
    min_delta : float
        Lower bound on the confidence gap between selected and discarded candidates.
    evaluate : Callable
        Function that takes a candidate np.ndarray and returns an integer reward.

    Returns
    -------
    List of the best *n* candidates.

    """
    num_candidates = len(candidates)
    if num_candidates <= n:
        raise ValueError(f"Less than {n} + 1 candidates!")

    m = num_candidates - n

    samples = np.ones(num_candidates, dtype=int)
    rewards = np.array([evaluate(candidate) for candidate in candidates], dtype=int)

    def draw(candidate_idx):
        rewards[candidate_idx] += evaluate(candidates[candidate_idx])
        samples[candidate_idx] += 1

    for _ in range(9):
        for idx in range(num_candidates):
            draw(idx)

    observed_means = rewards / samples

    lower_bound = np.array([compute_bound(num_samples=samples[i],
                                          observed_mean=observed_means[i],
                                          divergence_bound=1.0,
                                          previous_target_bound=0.0,
                                          previous_other_bound=1.0,
                                          goal=min) for i in range(num_candidates)])
    upper_bound = np.array([compute_bound(num_samples=samples[i],
                                          observed_mean=observed_means[i],
                                          divergence_bound=1.0,
                                          previous_target_bound=1.0,
                                          previous_other_bound=0.0,
                                          goal=max) for i in range(num_candidates)])

    def update(candidate_idx):
        logger.debug(f"Samples for candidate {candidate_idx}: {samples[candidate_idx]}")

        for bound, other_bound, goal in [(lower_bound, upper_bound, min), (upper_bound, lower_bound, max)]:
            logger.debug(f"Old {goal.__name__} bound: {bound[candidate_idx]}")
            lower_bound[candidate_idx] = compute_bound(num_samples=samples[candidate_idx],
                                                       observed_mean=rewards[candidate_idx] / samples[candidate_idx],
                                                       divergence_bound=1.0,
                                                       previous_target_bound=bound[candidate_idx],
                                                       previous_other_bound=other_bound[candidate_idx],
                                                       goal=goal)
            logger.debug(f"New {goal.__name__} bound: {bound[candidate_idx]}")

    for step in range(MAX_STEPS):
        observed_means = rewards / samples
        logger.debug(f"Observed means: Mean: {np.mean(observed_means):0.4f}, "
                     f"Min: {np.min(observed_means):0.4f}, "
                     f"Max: {np.max(observed_means):0.4f}")

        leading = np.argpartition(observed_means, -n)[-n:]
        losing = np.argpartition(observed_means, m)[:m]
        logger.debug(f"Observed leading means: Mean: {np.mean(observed_means[leading]):0.4f}, "
                     f"Min: {np.min(observed_means[leading]):0.4f}, "
                     f"Max: {np.max(observed_means[leading]):0.4f}")
        logger.debug(f"Observed losing means: Mean: {np.mean(observed_means[losing]):0.4f}, "
                     f"Min: {np.min(observed_means[losing]):0.4f}, "
                     f"Max: {np.max(observed_means[losing]):0.4f}")

        smallest_lower_bound = np.min(lower_bound[leading])
        largest_upper_bound = np.max(upper_bound[losing])

        logger.debug(f"Smallest lower bound: {smallest_lower_bound :0.4f}")
        logger.debug(f"Largest upper bound: {largest_upper_bound :0.4f}")

        if smallest_lower_bound - largest_upper_bound > min_delta:
            break

        leading_candidate_smallest_lower_bound_idx = leading[np.argmin(lower_bound[leading])]
        assert leading_candidate_smallest_lower_bound_idx in leading
        losing_candidate_largest_upper_bound_idx = losing[np.argmax(upper_bound[losing])]
        assert losing_candidate_largest_upper_bound_idx in losing

        draw(leading_candidate_smallest_lower_bound_idx)
        draw(losing_candidate_largest_upper_bound_idx)

        update(leading_candidate_smallest_lower_bound_idx)
        update(losing_candidate_largest_upper_bound_idx)
        logger.debug(step)
    else:
        logger.warning(f"{MAX_STEPS} steps exceeded")

    leading_candidates = np.argpartition(observed_means, -n)[-n:]
    losing_candidates = np.argpartition(observed_means, m)[:m]

    logger.debug(f"Rewards:\n- Leading candidates: {rewards[leading_candidates]}"
                 f"\n- Losing candidates: {rewards[losing_candidates]}")
    logger.debug(f"Means:\n- Leading candidates: {observed_means[leading_candidates]}"
                 f"\n- Losing candidates: {observed_means[losing_candidates]}")

    if np.any(rewards[leading_candidates] == 0):
        logger.warning("There are leading candidates with 0 rewards.")

    if np.any(observed_means[leading_candidates] == 0):
        logger.warning("There are leading candidates with mean 0.")

    return ([candidate for i, candidate in enumerate(candidates) if i in leading_candidates],
            np.array([float(lower_bound[i]) for i in leading_candidates]))
