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

    def generate_from_candidates(self, candidates: List[np.ndarray], num_children: int,
                                 objective: Union[Objective, None] = None,
                                 threshold: Union[float, None] = None,
                                 keep_parent: bool = False) -> List[np.ndarray]:
        """Generate new candidates by evolving the *candidates* according to the strategies's strategy.

        For each candidate, *num_children* new candidates are generated.

        If an *objective* is given, only candidates for which the objective function is below
        the specified *threshold* are returned.

        Parameters
        ----------
        candidates : List of np.ndarrays
            The candidates that seed the generation.
        num_children : int
            Number of descendants to generate for each candidate.
        objective : Callable, optional
            `Objective` instance to calculate the potential candidate's objective value.
        threshold : float, optional unless `objective` is given
            Only candidates with objective value below this threshold are returned.
        keep_parent : bool, optional
            If `True`, parent candidate is itself a child.

        Returns
        -------
        List of candidates.

        """
        if objective is None:
            objective = Zero()
            if threshold is not None:
                raise ValueError("If objective is not given, threshold must not be set.")
            threshold = 1.0
        else:
            objective.update()

        if threshold is None:
            threshold = 0.0

        children = []
        for candidate in candidates:
            _threshold = max(objective(candidate), threshold)
            _fit_children = []
            if keep_parent:
                _fit_children += [candidate]
            _children = self._generate_from_candidate(candidate, num_children)
            _fit_children += [child for child in _children
                              if objective(child) < _threshold and np.any(child)]
            children += _fit_children[:num_children]
        return children

    def _generate_from_candidate(self, candidate: np.ndarray, num_children: int) -> List[np.ndarray]:
        strategies = np.random.choice(self.strategies, size=num_children, replace=True, p=self.weights)
        return [strategy(candidate) for strategy in strategies]
