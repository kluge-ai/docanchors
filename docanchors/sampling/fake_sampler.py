import time
from multiprocessing import Queue
from queue import Full

import numpy as np

from .sampler import Sampler


class FakeSampler(Sampler):

    def __init__(self,
                 instance: np.ndarray,
                 sample_queue: Queue,
                 num_of_samples: int = 1000):
        super(FakeSampler, self).__init__()

        self.instance = instance
        self.sample_queue = sample_queue
        self.num_of_samples = num_of_samples

    def run(self):

        while True:
            if self.sample_queue.full():
                time.sleep(0.1)
                continue

            for _ in range(self.num_of_samples):
                try:
                    self.sample_queue.put((np.ones_like(self.instance), 1),
                                          block=True)
                except Full:
                    break
