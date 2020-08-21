from typing import List, Union

import numpy as np

from .strategies import Strategy, Seed
from .objectives import Objective, Zero
from ..util.configuration.component import Component


class Generator(Component):
    """Generator for candidates.

    Parameters
    ----------
    strategies : list of Strategy
        Strategies used to generate child candidates.
    weights : list of floats, optional
        Relative weight of each given strategy. For each candidate, one of the given
        strategies is randomly selected.
    seed : Objective, optional
        The `Objective` to generate the seed candidates with. Receives an all-zero
        candidate during candidate generation. Typically, instances of
        `objectives.Seed` are used. If not given, `Seed()` in default configuration
        is used.

    Examples
    --------

    """

    def __init__(self,
                 strategies: List[Strategy],
                 weights: Union[List[float], None] = None,
                 seed: Union[Strategy, None] = None):
        super(Generator, self).__init__()
        self.strategies = strategies
        weights = weights or [1.0 for _ in self.strategies]
        self.weights = np.array(weights) / np.sum(weights)
        self.seed = seed or Seed()

    def generate_candidates(self, size: int, num_candidates: int) -> List[np.ndarray]:
        """Generate *num_candidates* initial candidates to seed the search.

        Parameters
        ----------
        size : int
            Length of a candidate, usually equal to document length.
        num_candidates : int
            Number of candidates go generate.

        Returns
        -------
        List of candidates.

        """
        return [self.seed(np.zeros(size, dtype=bool)) for _ in range(num_candidates)]

    def generate_from_candidates(self,
                                 candidates: List[np.ndarray],
                                 num_children: int,
                                 force_randomization: bool = False,
                                 keep_parent: bool = False) -> List[np.ndarray]:
        """Generate new candidates by evolving the *candidates* according to the strategies's strategy.

        For each candidate, *num_children* new candidates are generated.

        Parameters
        ----------
        candidates : List of np.ndarrays
            The candidates that seed the generation.
        num_children : int
            Number of descendants to generate for each candidate.
        force_randomization : bool, optional
            If `True`, always randomly choose from strategies with replacement.
        keep_parent : bool, optional
            If `True`, parent candidate is itself a child.

        Returns
        -------
        List of candidates.

        """
        replace = force_randomization or num_children > len(self.strategies)

        children = []
        for candidate in candidates:
            if keep_parent:
                children.append(candidate)

            children += self._generate_from_candidate(candidate, num_children, replace)

        return children

    def _generate_from_candidate(self, candidate: np.ndarray, num_children: int, replace: bool = False):
        strategies = np.random.choice(self.strategies, size=num_children, replace=replace, p=self.weights)
        return [strategy(candidate) for strategy in strategies]

    def mutate_candidate(self, candidate: np.ndarray) -> np.ndarray:
        return self._generate_from_candidate(candidate, num_children=1)[0]

