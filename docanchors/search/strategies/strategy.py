from typing import Dict, Any

import numpy as np

from ...util.configuration.component import Component


class Strategy(Component):
    """Base class for search strategies.

    A `Strategy` is called on a `candidate` array returns a child,
    which generally is created by evolving the `candidate`.

    ..note::
        When implementing strategies, care must be taken to not
        modify the incoming `candidate`, but to create a copy
        or entirely new array. Otherwise, further `Strategy`
        calls on the `candidate` will evolve the newly created
        child instead of the original, disturbing the search.

    """

    def __init__(self, *args, **kwargs):
        super(Strategy, self).__init__()
        self._random = np.random.default_rng()

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        raise NotImplementedError
