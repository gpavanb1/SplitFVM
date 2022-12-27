from .domain import Domain
from .system import System
from .refine import Refiner
from .model import Model


class Simulation:
    def __init__(self, d: Domain, m: Model):
        self._d = d
        self._s = System(m)

    def evolve(self, dt: float, refinement: bool = False):
        # Fill BCs
        for x in self._d._components:
            self._d.apply_BC(x)

        # Evaluate residuals (values, face-values, fluxes) from equations
        interior_residual_block = self._s.residuals(self._d)

        # Update cell values and faces
        self._d.update(dt, interior_residual_block)

        # Perform mesh refinement if enabled
        if refinement:
            Refiner.refine(self._d)
