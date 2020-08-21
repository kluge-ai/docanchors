import random
import logging
import numpy as np
from typing import Callable

from .generator import Generator
from .objectives import Objective
from .kl_lucb import find_best_n


class LocalBeamSearch:

    def __init__(self, generator: Generator, objective: Objective, evaluate_fn: Callable[[np.ndarray], int]):
        self.logger = logging.getLogger(__name__)
        self.generator = generator
        self.objective = objective
        self._evaluate_candidate = evaluate_fn

        self.num_seed_candidates = 12
        self.num_children = 12
        self.num_candidates = 10

        self.num_elite_candidates = 2

        self.threshold_prefactor = 1.0
        self.max_steps = 10
        self.min_delta = 0.0
        self.patience = 3
        self.min_threshold = 5

    def find_best_anchor(self, instance: np.ndarray) -> np.ndarray:
        self.logger.debug("Generate initial candidates")
        candidates = self.generator.generate_candidates(len(instance),
                                                        num_candidates=self.num_seed_candidates)

        elite_candidates = []

        best_candidate = np.any(np.array(candidates), axis=0)
        best_objective_value = self.objective(best_candidate)

        early_stopping_counter = 0

        for t in range(self.max_steps):
            self.logger.debug(f"Step {t}")

            self.logger.debug(f"Generate new candidates")
            new_candidates = self.generator.generate_from_candidates(candidates,
                                                                     num_children=self.num_children)
            new_candidates += elite_candidates
            random.shuffle(new_candidates)

            self.logger.debug(f"Find the {self.num_candidates} best candidates")
            try:
                candidates, lower_bounds = find_best_n(new_candidates,
                                                       n=self.num_candidates,
                                                       min_delta=self.min_delta,
                                                       evaluate=self._evaluate_candidate)
            except ValueError:
                break

            candidates = [candidates[i] for i in np.argsort(-lower_bounds)]
            elite_candidates = [candidate.copy() for candidate in candidates[:self.num_elite_candidates]]

            self.logger.debug("Calculate objective function for all candidates")
            new_candidate = np.any(np.array(candidates), axis=0)

            if self.objective(new_candidate) < best_objective_value:
                best_candidate = new_candidate
                best_objective_value = self.objective(new_candidate)
                early_stopping_counter = 0
            else:
                self.logger.info("Failed to surpass best candidate")
                early_stopping_counter += 1
                if early_stopping_counter > self.patience:
                    break

            if best_objective_value < self.min_threshold:
                self.logger.debug(f"{best_objective_value} is below {self.min_threshold}")
                if early_stopping_counter <= self.patience:
                    continue
                else:
                    break

        self.logger.info(f"Finished search: {self.objective.log_value(best_candidate)}")
        return best_candidate
