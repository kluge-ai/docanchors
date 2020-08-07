import logging
from collections import deque
from multiprocessing import Queue
from typing import Dict, Any, Union

import numpy as np

from .search import LocalBeamSearch, GeneticSearch
from .search.generator import Generator
from .search.objectives import Objective
from .util.util import get_final_padding_length

# TODO: Logging infrastructure

EPSILON = 1e-6
MAXLOOP = 5


class DocumentAnchor:
    # TODO: Document all attributes
    """

    Parameters
    ----------
    sample_queue : Queue
        Queue for samples from the perturbation set.
    generator : Generator
        Anchor candidate generator.
    objective : Objective
        Evaluation objective.

    Attributes
    ----------

    """

    def __init__(self,
                 sample_queue: Queue,
                 generator: Generator,
                 objective: Objective,
                 search: Union[LocalBeamSearch, GeneticSearch]):
        self.sample_queue = sample_queue
        self.generator = generator
        self.objective = objective
        self.search = search

        self.match_condition = 0.80

        self.logger = logging.getLogger("DocumentAnchor")
        self.buffer = deque([], maxlen=10000)

    @classmethod
    def from_configuration(cls, configuration: Dict[str, Any]):
        # TODO: Load from text-based configuration
        # TODO: Figure out how to inject the queues
        pass

    def explain(self, instance: np.ndarray, restrict_to_known: bool = True) -> np.ndarray:
        """

        Parameters
        ----------
        instance : np.ndarray
            The document instance for which an anchor is sought.
        restrict_to_known : bool
            If *True*, restrict search for anchor to range of non-zero tokens.

        Returns
        -------
        best_candidate : np.ndarray
            The best document anchor found as a boolean array of the same size as `instance`.

        """
        if restrict_to_known:
            final_padding_length = get_final_padding_length(instance)
            _instance = instance[:-final_padding_length]
        else:
            _instance = instance

        best_candidate = self.search.find_best_anchor(_instance)

        if restrict_to_known:
            best_candidate = np.pad(best_candidate, (0, len(instance) - len(best_candidate)),
                                    'constant', constant_values=(0, 0))

        return best_candidate

    def evaluate_candidate(self, candidate: np.ndarray, num: int = 100) -> float:
        return float(np.mean([self._evaluate_candidate(candidate) for _ in range(num)]))

    def _evaluate_candidate(self, candidate: np.ndarray) -> int:
        # TODO: If advantageous, refactor to take samples as sparse matrices
        _sum_candidate = np.sum(candidate)
        if _sum_candidate == 0:
            return 0

        for _ in range(MAXLOOP):
            if self.sample_queue.empty() and len(self.buffer) > MAXLOOP:
                try:
                    sample, label = self.buffer[0]
                except IndexError:
                    self.logger.warning("Empty buffer, need to wait for sampler.")
                    sample, label = self.sample_queue.get()
                else:
                    self.buffer.rotate(-1)
            else:
                sample, label = self.sample_queue.get()
                self.buffer.append((sample, label))

            _sample = sample[:len(candidate)]
            match = np.sum(_sample * candidate) / min(_sum_candidate, np.sum(_sample))
            if match >= self.match_condition - EPSILON:
                return label
        else:
            # self.logger.warning(f"Candidate did not match for {MAXLOOP} samples. Low coverage.")
            return 0
