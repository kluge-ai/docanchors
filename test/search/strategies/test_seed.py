import numpy as np

from docanchors.search.strategies.seed import Seed


def test_seed_size(mocker):
    for size in range(10):
        candidate = np.zeros(20, dtype=bool)
        s = Seed(size=size, num_seeds=1)

        mocker.patch.object(s, "_random")
        s._random.choice.return_value = np.array([0])

        result = s(candidate)

        assert np.sum(result) == size
        assert np.sum(result[:size]) == size


def test_seed_position(mocker):
    for position in range(20):
        candidate = np.zeros(25, dtype=bool)
        s = Seed(size=1, num_seeds=1)

        mocker.patch.object(s, "_random")
        s._random.choice.return_value = np.array([position])

        result = s(candidate)

        assert np.sum(result) == 1
        assert result[position]


def test_seed_num(mocker):
    for number_of_seeds in range(5):
        candidate = np.zeros(30, dtype=bool)
        s = Seed(size=1, num_seeds=number_of_seeds)

        mocker.patch.object(s, "_random")
        s._random.choice.return_value = np.array([2 * i for i in range(number_of_seeds)])

        result = s(candidate)

        assert np.sum(result) == number_of_seeds
        assert np.all(result[[2 * i for i in range(number_of_seeds)]])
