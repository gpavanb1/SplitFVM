from .domain import Domain
from .system import System
from .refine import Refiner
from .model import Model
from .initialize import set_initial_condition


class Simulation:
    def __init__(self, d: Domain, m: Model, ics: dict, bcs: dict):
        self._d = d
        self._s = System(m)
        self._r = Refiner()
        self._bcs = bcs

        # Set initial conditions
        for c, ictype in ics.items():
            set_initial_condition(self._d, c, ictype)

    def evolve(self, dt: float, refinement: bool = False):
        # Fill BCs
        for c, bctype in self._bcs.items():
            self._d.apply_BC(c, bctype)

        # Evaluate residuals (values, face-values, fluxes) from equations
        interior_residual_block = self._s.residuals(self._d)

        # Update cell values and faces
        self._d.update(dt, interior_residual_block)

        # Perform mesh refinement if enabled
        if refinement:
            self._r.refine(self._d)
