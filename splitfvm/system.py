import numpy as np
from .domain import Domain
from .flux import Schemes


class System:
    """
    Construct residuals given various types of
    equations in the models
    """

    def __init__(self, model, scheme=Schemes.LF):
        self._model = model
        self._scheme = scheme

    def residuals(self, d: Domain):
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
