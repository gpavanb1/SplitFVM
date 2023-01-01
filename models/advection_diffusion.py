import numpy as np
from splitfvm.model import Model
from splitfvm.equations.transport import TransportEquation


class AdvectionDiffusion(Model):
    def __init__(self, c, nu):
        self.c = c
        self.nu = nu
        F = lambda u: np.array([self.c * x for x in u])
        D = lambda u: np.array([self.nu * x for x in u])
        S = lambda u: np.array([0.0])
        self._equations = [TransportEquation(F, D, S)]
