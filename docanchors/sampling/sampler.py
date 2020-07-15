import logging
from multiprocessing import Process


class Sampler:
    """Abstract base class for samplers."""

    def __init__(self):
        self.logger = logging.getLogger(self.__name__)

    def run(self):
        raise NotImplementedError


class MultiprocessingSampler(Process):
    """Run a `Sampler` in a separate process.

    Parameters
    ----------
    sampler : Sampler
    """

    def __init__(self, sampler: Sampler):
        super(MultiprocessingSampler, self).__init__()
        self.sampler = sampler

    def run(self):
        self.sampler.run()
