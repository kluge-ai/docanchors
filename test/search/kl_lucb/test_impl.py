import pytest

from docanchors.search.kl_lucb.impl_python import compute_bound as py_compute_bound
from docanchors.search.kl_lucb.impl_python import kl as py_kl
from docanchors.search.kl_lucb.impl_cython import compute_bound as cy_compute_bound
from docanchors.search.kl_lucb.impl_cython import kl as cy_kl

kwargs = {
    "num_samples": 1000,
    "observed_mean": 0.4,
    "divergence_bound": 0.01,
    "value": 0.25,
    "target_bound": 0.6,
    "precision": 1e-3
}


def test_equivalency():
    py_bound = py_compute_bound(**kwargs)
    cy_bound = cy_compute_bound(**kwargs)
    assert py_bound == pytest.approx(cy_bound, abs=kwargs["precision"])


def test_benchmark_compute_bound_py(benchmark):
    benchmark(py_compute_bound, **kwargs)


def test_benchmark_compute_bound_cy(benchmark):
    benchmark(cy_compute_bound, **kwargs)


def test_kullback_leibler(subtests):
    """Reference values taken from
    https://github.com/Naereen/Kullback-Leibler-divergences-and-kl-UCB-indexes/blob/master/src/kullback_leibler.py#L49
    """

    for impl, kl in [("python", py_kl), ("cython", cy_kl)]:
        for args, expected in [((0.5, 0.5), 0.0),
                               ((0.1, 0.9), 1.757779),
                               ((0.9, 0.1), 1.757779),
                               ((0.4, 0.5), 0.020135),
                               ((0.5, 0.4), 0.020411),
                               ((0.01, 0.99), 4.503217)]:
            with subtests.test(msg=f"Testing {impl} for {args}",
                               kl=kl, args=args, expected=expected):
                assert kl(*args) == pytest.approx(expected, abs=1e-5)
