import numpy as np
import jax
import numdifftools as nd
from splitnewton.newton import newton
from splitnewton.split_newton import split_newton

from .domain import Domain
from .error import SFVM
from .system import System
from .refine import Refiner
from .model import Model

# ICs and BCs
from .bc import apply_BC
from .initialize import set_initial_condition


def array_list_reshape(l, shape):
    # Reshape to list of 1D numpy arrays
    return [np.array(x) for x in np.reshape(l, shape).tolist()]


class Simulation:
    def __init__(self, d: Domain, m: Model, ics: dict, bcs: dict, ss: dict = {}):
        self._d = d
        self._s = System(m)
        self._r = Refiner()
        self._bcs = bcs

        # Steady-state solver settings
        self._ss = ss

        # Set initial conditions
        for c, ictype in ics.items():
            set_initial_condition(self._d, c, ictype)

        # Fill BCs
        for c, bctype in self._bcs.items():
            apply_BC(self._d, c, bctype)

    def evolve(self, dt: float, refinement: bool = False):
        # Fill BCs
        for c, bctype in self._bcs.items():
            apply_BC(self._d, c, bctype)

        # Evaluate residuals (values, face-values, fluxes) from equations
        interior_residual_block = self._s.residuals(self._d)

        # Update cell values and faces
        self._d.update(dt, interior_residual_block)

        # Perform mesh refinement if enabled
        if refinement:
            self._r.refine(self._d)

    ############
    # List related methods
    ############

    def get_shape_from_list(self, l):
        nv = self._d.num_components()

        if len(l) % nv != 0:
            raise SFVM("List length not aligned with interior size")

        num_points = len(l) // nv

        return num_points, nv

    def initialize_from_list(self, l, split=False, split_loc=None):
        # Just demarcate every nv entries as a block
        # This gives block-diagonal structure in unsplit case
        num_points, nv = self.get_shape_from_list(l)

        if split:
            # Same as SplitNewton convention
            # Outer system will be excluding `loc`
            outer_block = array_list_reshape(
                l[: split_loc * num_points], (-1, split_loc)
            )
            inner_block = array_list_reshape(
                l[split_loc * num_points :], (-1, nv - split_loc)
            )

            block = []
            for i in range(num_points):
                block.append(np.concatenate((outer_block[i], inner_block[i])))
        else:
            # Reshape list
            block = array_list_reshape(l, (num_points, nv))

        # Assign values to cells in domain
        cells = self._d.interior()
        for i, b in enumerate(cells):
            b.set_values(block[i])

    def get_residuals_from_list(self, l, split=False, split_loc=None):
        # Assign values from list
        # Note domain already exists and we preserve distances
        self.initialize_from_list(l, split, split_loc)

        # Fill BCs
        for c, bctype in self._bcs.items():
            apply_BC(self._d, c, bctype)

        interior_residual_block = self._s.residuals(self._d)

        if split:
            outer_block = [x[:split_loc] for x in interior_residual_block]
            inner_block = [x[split_loc:] for x in interior_residual_block]
            outer_list = np.array(outer_block).flatten()
            inner_list = np.array(inner_block).flatten()
            residual_list = np.concatenate((outer_list, inner_list))
        else:
            # Reshape residual block in list order
            residual_list = np.array(interior_residual_block).flatten()

        return residual_list

    def jacobian(self, l, split=False, split_loc=None):
        _f = lambda u: self.get_residuals_from_list(u, split, split_loc)
        solver_type = self._ss.get("jacobian_type", "numerical")

        if solver_type.lower() == "forward":
            return jax.jacfwd(_f)(l)
        elif solver_type.lower() == "reverse":
            return jax.jacrev(_f)(l)
        elif solver_type.lower() == "numerical":
            return nd.Jacobian(_f, method="central")(l)

    def steady_state(
        self, split=False, split_loc=None, sparse=True, dt0=0.0, dtmax=1.0, armijo=False
    ):
        _f = lambda u: self.get_residuals_from_list(u, split, split_loc)
        _jac = lambda u: self.jacobian(u, split, split_loc)

        x0 = self._d.listify_interior(split, split_loc)

        if not split:
            xf, _, iter = newton(
                _f, _jac, x0, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo
            )
        else:
            if split_loc is None:
                raise SFVM("Split location must be specified in this case")

            num_points, _ = self.get_shape_from_list(x0)
            loc = num_points * split_loc
            xf, _, iter = split_newton(
                _f, _jac, x0, loc, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo
            )

        self.initialize_from_list(xf, split, split_loc)
        return iter
