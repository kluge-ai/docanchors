import logging
import random
from collections import deque
from multiprocessing import Queue
from typing import Dict, Any

import numpy as np

from .search.generator import Generator
from .search.kl_lucb import find_best_n
from .search.objectives import Objective
from .util.util import get_final_padding_length


# TODO: Logging infrastructure

EPSILON = 1e-6
MAXLOOP = 50


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

    def __init__(self, sample_queue: Queue, generator: Generator, objective: Objective):
        self.generator = generator
        self.sample_queue = sample_queue
        self.objective = objective

        self.num_seed_candidates = 40
        self.num_candidates = 8

        self.max_steps = 500
        self.num_children = 3
        self.random_candidates = 0
        self.min_threshold = 0.1
        self.min_delta = 0.05
        self.threshold_prefactor = 1.0
        self.patience = 5
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
        self._check_parameters()

        if restrict_to_known:
            final_padding_length = get_final_padding_length(instance)
            _instance = instance[:-final_padding_length]
        else:
            _instance = instance

        best_candidate = self._find_anchor(_instance)

        if restrict_to_known:
            best_candidate = np.pad(best_candidate, (0, len(instance) - len(best_candidate)),
                                    'constant', constant_values=(0, 0))

        return best_candidate

    def _find_anchor(self, instance: np.ndarray) -> np.ndarray:
        self.logger.debug("Generate initial candidates")
        candidates = self.generator.generate_candidates(len(instance),
                                                        num_candidates=self.num_seed_candidates)

        self.objective.update()
        objective_values = [self.objective(candidate) for candidate in candidates]
        threshold = self.threshold_prefactor * np.max(objective_values)

        best_candidates = candidates
        best_objective_values = objective_values

        early_stopping_counter = 0

        for t in range(self.max_steps):
            self.logger.debug(f"Step {t}")

            self.logger.debug(f"Generate new candidates")
            new_candidates = self.generator.generate_from_candidates(candidates,
                                                                     num_children=self.num_children,
                                                                     objective=self.objective)

            for _ in range(self.num_children):
                self.logger.debug("Generate more candidates...")
                if len(new_candidates) < self.num_candidates * self.num_children:
                    new_candidates += self.generator.generate_from_candidates(new_candidates,
                                                                              num_children=1,
                                                                              objective=self.objective,
                                                                              threshold=threshold)
                else:
                    new_candidates = new_candidates[:self.num_candidates * self.num_children]
                    break
            else:
                self.logger.warning("Did not find enough candidates")
                if len(new_candidates) < self.num_candidates * self.num_children:
                    new_candidates += self.generator.generate_from_candidates(candidates,
                                                                              num_children=self.num_children,
                                                                              objective=self.objective,
                                                                              threshold=1.0)

            random.shuffle(new_candidates)

            self.logger.debug(f"Find the {self.num_candidates - self.random_candidates} best candidates")
            try:
                candidates, lower_bounds = find_best_n(new_candidates,
                                                       n=self.num_candidates - self.random_candidates,
                                                       min_delta=self.min_delta,
                                                       evaluate=self._evaluate_candidate)
            except ValueError:
                break

            candidates = [candidates[i] for i in np.argsort(-lower_bounds)]

            if self.random_candidates > 0:
                self.logger.debug(f"Add {self.random_candidates} randomly selected candidates")
                candidates.extend(random.sample(new_candidates, self.random_candidates))

            self.logger.debug("Calculate objective function for all candidates")
            self.objective.update()
            objective_values = [self.objective(candidate) for candidate in candidates]
            best_objective_values = [self.objective(candidate) for candidate in best_candidates]

            if min(objective_values) <= min(best_objective_values):
                best_candidates = candidates
                early_stopping_counter = 0
            else:
                self.logger.warning("Failed to surpass best candidate")
                early_stopping_counter += 1
                if early_stopping_counter > self.patience:
                    break

            self.logger.debug(f"Current value: {min(objective_values)} (Previous Best: {min(best_objective_values)}")
            if min(objective_values) < self.min_threshold:
                self.logger.debug(f"{min(objective_values)} is below {self.min_threshold}")
                if t < self.patience:
                    continue
                else:
                    break
            else:
                self.logger.debug(f"{min(objective_values)} is NOT below {self.min_threshold}")
                threshold = self.threshold_prefactor * np.max(objective_values)
        else:
            self.logger.warning(f"Failed to generate {self.num_candidates} new candidates below current threshold "
                                f"{threshold} (pre-factor {self.threshold_prefactor}) in {self.max_steps} steps.")

        best_candidate = best_candidates[int(np.argmin(best_objective_values))]

        self.logger.info(f"Finished search: {self.objective.log_value(best_candidate)}")
        return best_candidate

    def evaluate_candidate(self, candidate: np.ndarray, num: int = 100) -> float:
        return float(np.mean([self._evaluate_candidate(candidate) for _ in range(num)]))

    def _evaluate_candidate(self, candidate: np.ndarray) -> int:
        # TODO: If advantageous, refactor to take samples as sparse matrices
        _sum_candidate = np.sum(candidate)
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
            self.logger.warning(f"Candidate did not match for {MAXLOOP} samples. Low coverage.")
            return 0

    def _check_parameters(self):
        if self.num_children >= (self.num_candidates - self.random_candidates):
            self.logger.warning(f"Number of children per candidate ({self.num_children}) is equal to "
                                f"or exceeds number of best candidates chosen in each round "
                                f"({self.num_candidates - self.random_candidates}. This might "
                                f"overly restrict the beam search to closely related beams.")

        if self.random_candidates > self.num_candidates:
            raise ValueError("Number of random candidates must not exceed number of candidates.")

        if self.random_candidates == self.num_candidates:
            self.logger.warning("Number of random candidates equals number of candidates. Beam search is "
                                "fully stochastic.")
