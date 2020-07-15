from typing import List

from .component import Component


def test_initialization():
    c = Component()
    assert len(c.configuration) == 0


def test_subclassing():
    class Some(Component):
        pass

    assert len(Some().configuration) == 0


def test_that_arguments_are_used():
    class Some(Component):

        def __init__(self, value: int):
            super(Some, self).__init__()
            self.value = value

    config = Some(5).configuration

    assert len(config) == 1
    assert config == {"value": 5}

    some = Some.from_configuration(config)

    assert some.value == 5


def test_that_hidden_attributes_are_ignored():
    class Some(Component):

        def __init__(self):
            super(Some, self).__init__()
            self._batch = [1, 2, 3, 4]

    config = Some().configuration

    assert len(config) == 0


def test_that_unused_attributes_are_ignored():
    class Some(Component):

        def __init__(self):
            super(Some, self).__init__()
            self.batch = [1, 2, 3, 4]

    config = Some().configuration

    assert len(config) == 0


def test_that_kwargs_are_passed_to_parent():
    class SomeParent(Component):

        def __init__(self, pre_factor: float = 1.0):
            super(SomeParent, self).__init__()
            self.pre_factor = pre_factor

    class SomeChild(SomeParent):

        def __init__(self, **kwargs):
            super(SomeChild, self).__init__(**kwargs)

    config = SomeChild().configuration

    assert len(config) == 1
    assert config == {"pre_factor": 1.0}

    some_child = SomeChild.from_configuration(config)

    assert some_child.pre_factor == 1.0


def test_that_parents_unused_and_hidden_attributes_are_ignored():
    class SomeParent(Component):

        def __init__(self, pre_factor: float = 1.0):
            super(SomeParent, self).__init__()
            self.pre_factor = pre_factor
            self._hidden = "nothing"
            self.batch = [1, 2, 3, 4]

    class SomeChild(SomeParent):

        def __init__(self, **kwargs):
            super(SomeChild, self).__init__(**kwargs)

    config = SomeChild().configuration

    assert len(config) == 1
    assert config == {"pre_factor": 1.0}

    some_child = SomeChild.from_configuration(config)

    assert some_child.pre_factor == 1.0


def test_that_child_components_are_handled():
    class Sub(Component):

        def __init__(self, value: int):
            super(Sub, self).__init__()
            self.value = value

    class Some(Component):

        def __init__(self, children: List[Sub], child: Sub):
            super(Some, self).__init__()
            self.children = children
            self.child = child

    config = Some([Sub(4), Sub(8)], Sub(3)).configuration

    assert len(config) == 2
    assert len(config["children"]) == 2
    assert config["children"] == [{"component": "Sub", "config": {"value": 4}},
                                  {"component": "Sub", "config": {"value": 8}}]
    assert len(config["child"]) == 1
    assert config["child"] == [{"component": "Sub", "config": {"value": 3}}]

    some = Some.from_configuration(config)

    assert len(some.children) == 2
    assert isinstance(some.child, Sub)
    assert some.child.value == 3
