import numpy as np
from splitfvm.model import Model
from splitfvm.equations.transport import TransportEquation


class Advection(Model):
    def __init__(self, c):
        self.c = c
        F = lambda u: np.array([self.c * x for x in u])
        S = lambda u: np.array([0.0])
        self._equations = [TransportEquation(F, 0, S)]
