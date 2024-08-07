import numpy as np
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
    """
    Reshape a list into a list of 1D NumPy arrays.

    Parameters
    ----------
    l : list
        The list to reshape.
    shape : tuple
        The shape to reshape the list into.

    Returns
    -------
    reshaped_list : list of ndarray
        The reshaped list.
    """

    # Reshape to list of 1D numpy arrays
    return [np.array(x) for x in np.reshape(l, shape).tolist()]


class Simulation:
    """
    A class representing a simulation.

    Parameters
    ----------
    d : Domain
        The domain on which to perform the simulation.
    m : Model
        The model to use in the simulation.
    ics : dict
        The initial conditions to apply to the domain.
    bcs : dict
        The boundary conditions to apply to the domain.
    ss : dict, optional
        The steady-state solver settings to use in the simulation.
    """

    def __init__(self, d: Domain, m: Model, ics: dict, bcs: dict, ss: dict = {}):
        """
        Initialize a Simulation object.
        """
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
        """
        Evolve the simulation for a given time step.

        Parameters
        ----------
        dt : float
            The time step for the evolution.
        refinement : bool, optional
            Whether to perform mesh refinement. Defaults to False.
        """

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
        """
        Get the shape of the list when reshaped into a NumPy array.

        Parameters
        ----------
        l : list
            The list to get the shape for.

        Returns
        -------
        num_points : int
            The number of points in the reshaped array.
        nv : int
            The number of components per point in the reshaped array.
        """

        nv = self._d.num_components()

        if len(l) % nv != 0:
            raise SFVM("List length not aligned with interior size")

        num_points = len(l) // nv

        return num_points, nv

    def initialize_from_list(self, l, split=False, split_loc=None):
        """
        Initialize the domain from a list of values.

        Parameters
        ----------
        l : list
            The list of values to initialize the domain with.
        split : bool, optional
            Whether to split the values into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the values at. Required if `split` is True.
        """

        # Just demarcate every nv entries as a block
        # This gives block-diagonal structure in unsplit case
        num_points, nv = self.get_shape_from_list(l)

        if split:

            if split_loc is None:
                raise SFVM("Split location must be specified in this case")

            # Same as SplitNewton convention
            # Outer system will be excluding `loc`
            outer_block = array_list_reshape(
                l[: split_loc * num_points], (-1, split_loc)
            )
            inner_block = array_list_reshape(
                l[split_loc * num_points:], (-1, nv - split_loc)
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
        """
        Get the residuals for the domain given a list of values.

        Parameters
        ----------
        l : list
            The list of values to get the residuals for.
        split : bool, optional
            Whether to split the residuals into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the residuals at. Required if `split` is True.

        Returns
        -------
        residual_list : list
            The list of residual values.
        """

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

    def extend_bounds(self, bounds, num_points, nv, split=False, split_loc=None):
        """
        Extends the provided input bounds based on whether there is a split or not.

        Parameters:
        ----------
        bounds : list of list
            A list containing two lists, each of size nv, representing the lower and upper bounds.
        num_points : int
            The number of points to extend each bound to.
        nv : int
            The number of variables, indicating the length of each bound list.
        split : bool, optional
            A flag indicating whether to split the bounds at a specific location. Default is False.
        split_loc : int, optional
            The index at which to split the bounds if split is True. Default is None.

        Returns:
        -------
        list of list
            A list containing the extended lower and upper bounds.
        """
        # Check if bounds is a 2-list, each of size nv
        if len(bounds) != 2:
            raise SFVM("Bounds must be a list of 2 lists")
        else:
            if len(bounds[0]) != nv or len(bounds[1]) != nv:
                raise SFVM(
                    "Each list in bounds must be of length - number of variables")

        if not split:
            return [bounds[0] * num_points, bounds[1] * num_points]
        else:
            if split_loc is None:
                raise SFVM("split_loc must be provided if split is True")
            return [bounds[0][:split_loc] * num_points + bounds[0][split_loc:] * num_points,
                    bounds[1][:split_loc] * num_points + bounds[1][split_loc:] * num_points]

    def jacobian(self, l, split=False, split_loc=None):
        """
        Calculate the Jacobian of the system.

        Parameters
        ----------
        l : list
            The list of values to calculate the Jacobian for.
        split : bool, optional
            Whether to split the Jacobian into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the Jacobian at. Required if `split` is True.

        Returns
        -------
        jac : numpy.ndarray
            The Jacobian of the system.
        """

        def _f(u): return self.get_residuals_from_list(u, split, split_loc)
        return nd.Jacobian(_f, method="central")(l)

    def steady_state(
        self, split=False, split_loc=None, sparse=True, dt0=0.0, dtmax=1.0, armijo=False, bounds=None
    ):
        """
        Solve for the steady state of the system.

        Parameters
        ----------
        split : bool, optional
            Whether to split the solution into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the solution at. Required if `split` is True.
        sparse : bool, optional
            Whether to use a sparse Jacobian. Defaults to True.
        dt0 : float, optional
            The initial time step to use in pseudo-time. Defaults to 0.0.
        dtmax : float, optional
            The maximum time step to use  in pseudo-time. Defaults to 1.0.
        armijo : bool, optional
            Whether to use the Armijo rule for line searches. Defaults to False.

        Returns
        -------
        iter : int
            The number of iterations performed.
        """

        def _f(u): return self.get_residuals_from_list(u, split, split_loc)
        def _jac(u): return self.jacobian(u, split, split_loc)

        x0 = self._d.listify_interior(split, split_loc)
        num_points, nv = self.get_shape_from_list(x0)

        # Extend bounds based on input
        ext_bounds = self.extend_bounds(
            bounds, num_points, nv, split, split_loc)

        if not split:
            xf, _, iter = newton(
                _f, _jac, x0, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo,
                bounds=ext_bounds)
        else:
            if split_loc is None:
                raise SFDM("Split location must be specified in this case")

            loc = num_points * split_loc
            xf, _, iter = split_newton(
                _f, _jac, x0, loc, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo, bounds=ext_bounds
            )

        self.initialize_from_list(xf, split, split_loc)
        return iter
