import numpy as np
import logging
import random

from docanchors.search.kl_lucb.kl_lucb import find_best_n

logging.basicConfig(level=logging.DEBUG)


def test_minimal_case():
    winning_candidate = np.array([1, 0, 0])
    middle_candidate = np.array([1, 1, 0])
    losing_candidate = np.array([0, 0, 1])

    mask = np.array([1, 0, 0])

    def evaluate(candidate):
        return np.sum(candidate * mask) / np.sum(candidate)

    candidates = [middle_candidate, losing_candidate, winning_candidate]

    best_candidates = find_best_n(candidates,
                                  n=1, min_delta=0.1,
                                  evaluate=evaluate)

    assert np.alltrue(best_candidates[0] == winning_candidate)


def test_large_case():

    leading_candidates = [np.ones(shape=(100,)) for _ in range(10)]
    losing_candidates = [np.random.randint(low=0, high=1, size=100) for _ in range(10)]

    def evaluate(candidate):
        _mask = np.random.random(100) > 0.5
        return int(np.mean(candidate[_mask]) > 0.2)

    candidates = leading_candidates + losing_candidates
    random.shuffle(candidates)

    best_candidates = find_best_n(candidates,
                                  n=10, min_delta=0.1,
                                  evaluate=evaluate)


