import numpy as np
from docanchors.util.util import get_final_padding_length


class RandomAnchor:

    def __init__(self, target_coverage: int):
        self.target_coverage = target_coverage

    def explain(self, instance: np.ndarray, restrict_to_known: bool = True) -> np.ndarray:
        if restrict_to_known:
            final_padding_length = get_final_padding_length(instance)
            _instance = instance[:-final_padding_length]
        else:
            _instance = instance

        indices = np.random.choice(np.array(range(len(_instance))), size=self.target_coverage)
        best_candidate = np.zeros_like(instance, dtype=bool)
        best_candidate[indices] = True

        if restrict_to_known:
            best_candidate = np.pad(best_candidate, (0, len(instance) - len(best_candidate)),
                                    'constant', constant_values=(0, 0))

        return best_candidate

    def evaluate_candidate(self, candidate: np.ndarray, num: int = 100) -> float:
        return 0.0
