import numpy as np


class Cell:
    def __init__(self, x=None, value=np.array([])):
        self._value = value
        # X co-ordinate
        self._x = x

        # AMR deletion flag
        self.to_delete = False

    def x(self):
        return self._x

    def values(self):
        return self._value

    def value(self, i: int):
        return self._value[i]

    def set_value(self, i: int, val):
        self._value[i] = val

    def update(self, dt, residual):
        self._value -= dt * residual
