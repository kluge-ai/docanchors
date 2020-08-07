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
        self.random_candidates = 2

        self.threshold_prefactor = 1.0
        self.max_steps = 10
        self.min_delta = 0.0
        self.patience = 3
        self.min_threshold = 5

    def find_best_anchor(self, instance: np.ndarray) -> np.ndarray:
        self._check_parameters()

        self.logger.debug("Generate initial candidates")
        candidates = self.generator.generate_candidates(len(instance),
                                                        num_candidates=self.num_seed_candidates)

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
