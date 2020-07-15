import numpy as np

from docanchors.sampling.masking import RandomUniformMask, RandomChunkMask, RandomNonUniformMask


def test_zero_perturbation_rate(subtests):

    for rmask in [RandomUniformMask, RandomChunkMask]:
        with subtests.test(rmask=rmask):
            m = rmask()
            m.perturbation_rate = 0.0

            mask = m(batch_shape=(10, 100))

            assert np.all(mask)


def test_nonzero_perturbation_rate(subtests):

    for rmask in [RandomUniformMask, RandomChunkMask]:
        for rate in [0.25, 0.5, 0.75]:
            with subtests.test(rmask=rmask, rate=rate):
                m = rmask()
                m.perturbation_rate = rate

                mask = m(batch_shape=(64, 1024))
                assert np.isclose(np.mean(mask), 1.0 - m.perturbation_rate, rtol=0.02)
