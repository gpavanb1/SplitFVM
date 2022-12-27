import numpy as np
from enum import Enum
from .error import SFVM

btype = Enum("btype", "LEFT RIGHT")


class Boundary:
    def __init__(
        self, x, _btype, value=np.array([]), xmin: float = 0.0, xmax: float = 1.0
    ):
        # Check if correct type specified
        if x < xmin and _btype == btype.RIGHT or x > xmax and _btype == btype.LEFT:
            raise SFVM("Inappropriate boundary type given")

        self._value = value
        self._type = _btype
        # X co-ordinate
        self._x = x

    def x(self):
        return self._x

    def values(self):
        return self._value

    def value(self, i: int):
        return self._value[i]

    def set_value(self, i: int, val):
        self._value[i] = val
