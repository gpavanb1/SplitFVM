import numpy as np
from .domain import Domain
from .flux import Schemes


class System:
    """
    A class representing a system of equations.

    Parameters
    ----------
    model : Model
        The model for which to solve the system of equations.
    scheme : Schemes, optional
        The discretization scheme to use for the system. Defaults to Schemes.LF.
    """

    def __init__(self, model, scheme=Schemes.LF):
        """
        Initialize a System object.
        """
        self._model = model
        self._scheme = scheme

    def residuals(self, d: Domain):
        """
        Calculate the residuals for the system of equations.

        Parameters
        ----------
        d : Domain
            The domain for which to calculate the residuals.

        Returns
        -------
        rhs_list : list of ndarray
            The list of residual arrays for each cell in the domain.
        """

        cells = d.cells()

        # Interior indices
        ilo = d.ilo()
        ihi = d.ihi()

        rhs_list = []

        for i in range(ilo, ihi + 1):
            rhs = np.array([])
            # Append residuals from each equation
            for eq in self._model.equations():
                # Send two-sided stencil
                # Let model decide computation
                # Flux function specifies west or east
                cell_sub = [cells[i + offset] for offset in range(-d.nb(), d.nb() + 1)]
                rhs = np.concatenate((rhs, eq.residuals(cell_sub, self._scheme)))
            rhs_list.append(rhs)

        return rhs_list
