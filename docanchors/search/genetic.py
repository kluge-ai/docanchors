import random
import logging
import numpy as np
from typing import Callable, List

from .generator import Generator
from .objectives import Objective
from .kl_lucb import find_best_n


class GeneticSearch:

    def __init__(self, generator: Generator, objective: Objective, evaluate_fn: Callable[[np.ndarray], int]):
        self.logger = logging.getLogger(__name__)
        self.generator = generator
        self.objective = objective
        self._evaluate_candidate = evaluate_fn

        self.max_steps = 100
        self.parents_per_candidate = 2
        self.num_candidates = 100
        self.min_delta = 0.1
        self.mutation_rate = 0.5
        self.elite_size = 10
        self.min_threshold = 0

    def find_best_anchor(self, instance: np.ndarray) -> np.ndarray:
        # 0) Generate Seeds
        candidates = self.generator.generate_candidates(size=len(instance),
                                                        num_candidates=self.num_candidates)
        objective_values = [self.objective(candidate) for candidate in candidates]

        best_candidates = [candidates[i] for i in np.argsort(objective_values)[:self.elite_size]]

        for t in range(self.max_steps):
            # 1) Make new candidates
            p = 1 - np.array(objective_values) / np.sum(objective_values)

            candidates = [self.crossover(np.random.default_rng().choice(candidates, size=self.parents_per_candidate,
                                                                        replace=False, p=p/np.sum(p), axis=0))
                          for _ in range(self.parents_per_candidate * self.num_candidates)]

            # 2) Find best candidates
            candidates, lower_bounds = find_best_n(candidates,
                                                   n=self.num_candidates,
                                                   min_delta=self.min_delta,
                                                   evaluate=self._evaluate_candidate)
            candidates = [candidates[i] for i in np.argsort(-lower_bounds)]
            self.logger.info(f"{t} Lower bounds {lower_bounds[np.argsort(-lower_bounds)][:10]}")

            # 3) Score candidates with objective
            candidates = candidates + best_candidates
            objective_values = [self.objective(candidate) for candidate in candidates]

            keep = np.argsort(objective_values)[:self.num_candidates]
            candidates = [candidates[i] for i in keep]
            objective_values = [objective_values[i] for i in keep]

            self.logger.info(f"{t} Objective Values {objective_values[:10]}")

            best_candidates = candidates[:self.elite_size]

            if np.min(objective_values) < self.min_threshold:
                break

        # print(best_candidates)

        return best_candidates[0]

    def crossover(self, candidates: List[np.ndarray]) -> np.ndarray:
        new_candidate = np.zeros_like(candidates[0], dtype=bool)

        crossover_indices = [0]
        crossover_indices.extend(np.random.choice(range(len(new_candidate)), size=len(candidates) - 1))
        crossover_indices.append(len(new_candidate))

        for i, candidate in enumerate(candidates):
            if random.random() < self.mutation_rate:
                candidate = self.generator.mutate_candidate(candidate)
            # new_candidate += candidate
            new_candidate[crossover_indices[i]:crossover_indices[i + 1]] += candidate[crossover_indices[i]:crossover_indices[i + 1]]

        return new_candidate
