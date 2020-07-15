import inspect
from typing import Dict, Any, List

import numpy as np

from .resolver import resolve


class Component:

    def __init__(self, *args, **kwargs):
        pass

    @property
    def configuration(self) -> Dict[str, Any]:
        config = {}

        attributes = {member: value for member, value in self.__dict__.items()
                      if member[0] != "_" and not inspect.ismethod(value)}

        signature = inspect.signature(self.__class__)

        for name, value in signature.parameters.items():
            if name == "kwargs":
                for cls in inspect.getmro(self.__class__):
                    if cls is self.__class__:
                        continue

                    if cls is Component:
                        break

                    class_signature = inspect.signature(cls)
                    for _name, _value in class_signature.parameters.items():
                        if _name in attributes:
                            config[_name] = attributes[_name]

                    if "kwargs" not in class_signature.parameters.values():
                        break
            elif name == "args":
                continue
            else:
                config[name] = attributes[name]

        for name, value in config.items():
            if isinstance(value, Component):
                config[name] = [{"component": value.__class__.__name__,
                                 "config": value.configuration}]

            elif isinstance(value, tuple) or isinstance(value, list):
                if value and isinstance(value[0], Component):
                    config[name] = [{"component": v.__class__.__name__,
                                     "config": v.configuration}
                                    for v in value]

            # TODO: Do not store empty configs

            elif isinstance(value, np.ndarray):
                config[name] = value.tolist()

        return config

    @classmethod
    def from_configuration(cls, config: Dict[str, Any]):
        signature = inspect.signature(cls)

        arguments = {}

        for name, value in signature.parameters.items():
            if name in ("args", "kwargs"):
                continue

            if name in config:
                # TODO: Deal with missing configs for components
                if isinstance(config[name], list):
                    if isinstance(value.annotation, type(List)):
                        if value.annotation._name in ("List", "Tuple"):
                            arguments[name] = [resolve(c["component"], Component).from_configuration(c["config"]) for
                                               c in config[name]]

                            if value.annotation._name == "Tuple":
                                arguments[name] = tuple(arguments[name])
                    else:
                        if issubclass(value.annotation, Component):
                            c = config[name][0]
                            arguments[name] = resolve(c["component"], Component).from_configuration(c["config"])
                        else:
                            raise ValueError(
                                f"Invalid configuration. "
                                f"Class {cls.__name__} expects argument of type {str(value.annotation)}. "
                                f"Failed to resolve {config[name][0]} to match type.")

                else:
                    try:
                        arguments[name] = value.annotation(config[name])
                    except ValueError:
                        raise ValueError(
                            f"Invalid configuration. "
                            f"Class {cls.__name__} expects argument of type {str(value.annotation)}. "
                            f"Failed to convert {config[name]} to match type.")

        if "kwargs" in signature.parameters:
            kwargs = {name: value for name, value in config.items()
                      if name not in arguments}
            return cls(**arguments, **kwargs)
        else:
            return cls(**arguments)
