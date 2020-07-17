import logging
from multiprocessing import Queue
import numpy as np

from docanchors import DocumentAnchor
from docanchors.sampling import BatchSampler, MultiprocessingSampler
from docanchors.search.generator import Generator
from docanchors.search.objectives import AbsoluteCover, Coherence
from docanchors.search.strategies.highlight import Grow, Shift

logging.basicConfig(level=logging.INFO)

# Prepare fake instance and model interface
instance = np.random.randint(0, 100, size=25)


def get_predict_fn():
    """Load model from disk or prepare API calls."""

    def fake_model_predict(x: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 2, size=x.shape)

    return fake_model_predict


# Set up sampling component
sample_queue = Queue(maxsize=10000)

batch_sampler = MultiprocessingSampler(BatchSampler(instance=instance,
                                                    get_predict_fn=get_predict_fn,
                                                    sample_queue=sample_queue,
                                                    target=1))

# Set up search component
generator = Generator(strategies=[Grow(), Shift()])

objective = Coherence() + 2.0 * AbsoluteCover(target=5)

doc_anchor = DocumentAnchor(sample_queue=sample_queue,
                            generator=generator,
                            objective=objective)

if __name__ == "__main__":
    # Start sampler
    batch_sampler.start()

    # Start search
    anchor = doc_anchor.explain(instance)

    # Terminate sampler
    batch_sampler.terminate()

    print(anchor)

