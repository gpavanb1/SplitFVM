import numpy as np


class Cell:
    def __init__(self, x=None, value=np.array([])):
        self._value = value
        # X co-ordinate
        self._x = x

        # AMR deletion flag
        self.to_delete = False

    # Creating operators for sorting
    def __eq__(self, other):
        return self._x == other._x

    def __lt__(self, other):
        return self._x < other._x

    def x(self):
        return self._x

    def values(self):
        return self._value

    def value(self, i: int):
        return self._value[i]

    def set_value(self, i: int, val):
        self._value[i] = val

    def set_values(self, l):
        self._value = l

    # Note boundary does not have update
    def update(self, dt, residual):
        self._value += dt * residual
