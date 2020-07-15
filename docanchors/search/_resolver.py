from . import strategies
from . import objectives
from .strategies.strategy import Strategy
from .objectives.objectives import Objective


def resolve_strategy(name: str) -> Strategy:
    return strategies.__dict__[name]


def resolve_objective(name: str) -> Objective:
    return objectives.__dict__[name]
