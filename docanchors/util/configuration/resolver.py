from typing import Dict


def _find_subclasses(cls: type) -> Dict[str, type]:
    subclasses = {}
    for subclass in cls.__subclasses__():
        subclasses[subclass.__name__] = subclass
        if subclass.__subclasses__():
            subclasses.update(_find_subclasses(subclass))
    return subclasses


def resolve(name: str, cls: type):
    known_components = _find_subclasses(cls)

    try:
        return known_components[name]
    except KeyError:
        pass

    try:
        return globals()[name]
    except KeyError:
        raise LookupError(f"Cannot find component {name}")
