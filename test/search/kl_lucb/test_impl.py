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
    assert py_bound == pytest.approx(cy_bound, kwargs["precision"])


def test_benchmark_compute_bound_py(benchmark):
    benchmark(py_compute_bound, **kwargs)


def test_benchmark_compute_bound_cy(benchmark):
    benchmark(cy_compute_bound, **kwargs)

