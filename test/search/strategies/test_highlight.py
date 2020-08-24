import numpy as np

from docanchors.search.strategies.highlight import Grow, Shrink, Shift, Pass, Erase, ShiftLeft, ShiftRight


ALL_STRATEGIES = [Grow, Shrink, Pass, Erase, Shift, ShiftLeft, ShiftRight]


def test_that_empty_candidate_remains_unaltered(subtests):
    candidate = np.zeros(shape=(512,), dtype=bool)

    for strategy in ALL_STRATEGIES:
        with subtests.test(msg=f"{strategy.__name__} keeps empty candidate unaltered",
                           strategy=strategy):
            assert np.array_equal(strategy()(candidate), candidate)


def test_that_children_are_new_objects(subtests):
    candidate = np.random.randint(0, 2, size=256, dtype=bool)

    for strategy in ALL_STRATEGIES:
        with subtests.test(msg=f"{strategy.__name__} returns new object as child",
                           strategy=strategy):
            assert strategy()(candidate) is not candidate


def test_pass():
    candidate = np.random.randint(0, 2, size=256, dtype=bool)
    assert np.array_equal(Pass()(candidate), candidate)


def test_erase():
    candidate = np.random.randint(0, 2, size=256, dtype=bool)
    assert not np.any(np.nonzero(Erase()(candidate)))


def test_shrink_single_highlight_at_beginning():
    candidate = np.array([True, True, True, False, False])
    assert np.array_equal(Shrink()(candidate), np.array([True, True, False, False, False]))


def test_shrink_single_highlight_in_the_middle():
    candidate = np.array([False, True, True, False, False])
    assert np.array_equal(Shrink()(candidate), np.array([False, True, False, False, False]))


def test_shrink_single_highlight_at_the_end():
    candidate = np.array([False, False, False, True, True])
    assert np.array_equal(Shrink()(candidate), np.array([False, False, False, True, False]))


def test_shrink_multiple_highlights_at_the_beginning_and_the_end():
    candidate = np.array([True, True, True, False, False, False, False, False, True, True])
    assert np.array_equal(Shrink()(candidate),
                          np.array([True, True, False, False, False, False, False, False, True, False]))


def test_shrink_multiple_highlights_in_the_middle():
    candidate = np.array([False, True, True, False, False, True, True, False, False, False])
    assert np.array_equal(Shrink()(candidate),
                          np.array([False, True, False, False, False, True, False, False, False, False]))


def test_shrink_multiple_highlights_everywhere():
    candidate = np.array([True, True, True, False, False, True, True, False, True, True])
    assert np.array_equal(Shrink()(candidate),
                          np.array([True, True, False, False, False, True, False, False, True, False]))


def test_grow_single_highlight_at_beginning():
    candidate = np.array([True, True, True, False, False])
    assert np.array_equal(Grow()(candidate), np.array([True, True, True, True, False]))


def test_grow_single_highlight_in_the_middle():
    candidate = np.array([False, True, True, False, False])
    assert np.array_equal(Grow()(candidate), np.array([False, True, True, True, False]))


def test_grow_single_highlight_at_end():
    candidate = np.array([False, False, True, True, True])
    assert np.array_equal(Grow()(candidate), np.array([False, False, True, True, True]))


def test_grow_multiple_highlights_in_the_middle():
    candidate = np.array([False, True, True, False, False, True, True, False, False, False])
    assert np.array_equal(Grow()(candidate),
                          np.array([False, True, True, True, False, True, True, True, False, False]))


def test_grow_multiple_highlights_everywhere():
    candidate = np.array([True, True, False, False, True, True, False, False, True])
    assert np.array_equal(Grow()(candidate),
                          np.array([True, True, True, False, True, True, True, False, True]))


def test_grow_everything_is_highlighted():
    candidate = np.ones(10, dtype=bool)
    assert np.array_equal(Grow()(candidate), candidate)


def test_shift_left():
    candidate = np.array([False, False, True, True, True])
    shift = ShiftLeft()

    assert np.array_equal(shift(candidate), np.array([False, True, True, True, False]))


def test_shift_right():
    candidate = np.array([False, True, True, True, False])
    shift = ShiftRight()

    assert np.array_equal(shift(candidate), np.array([False, False, True, True, True]))


def test_shift_shifts_left(mocker):
    candidate = np.array([False, False, True, True, True])
    shift = Shift()

    mocker.patch.object(shift, "_random")
    shift._random.random.return_value = 0.1

    assert np.array_equal(shift(candidate), np.array([False, True, True, True, False]))


def test_shift_shifts_right(mocker):
    candidate = np.array([False, True, True, True, False])
    shift = Shift()

    mocker.patch.object(shift, "_random")
    shift._random.random.return_value = 0.7

    assert np.array_equal(shift(candidate), np.array([False, False, True, True, True]))
