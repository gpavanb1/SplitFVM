from numpy import linspace, zeros
from .boundary import btype, Boundary
from .cell import Cell
from .error import SFVM


class Domain:
    def __init__(
        self, cells: list[Cell], boundaries: list[Boundary], components: list[str]
    ):
        # Boundaries list contains both left and right
        # nb indicates number on each side
        self._nb = int(len(boundaries) / 2)
        self._nx = len(cells)
        self._domain = [*(boundaries[: self._nb]), *cells, *(boundaries[self._nb :])]
        self._components = components

    @classmethod
    def from_size(
        cls,
        nx: int,
        ng: int,
        components: list[str],
        xmin: float = 0.0,
        xmax: float = 1.0,
    ):
        if ng % 2 != 0:
            raise SFVM("nb must be an even number")

        # Initialize a uniform grid
        xarr = linspace(xmin, xmax, nx)
        nv = len(components)
        interior = [Cell(x, zeros(nv)) for x in xarr]

        # Create boundaries
        dx = (xmax - xmin) / nx
        left_boundaries = [
            Boundary(xmin - (i + 1) * dx, btype.LEFT, zeros(nv))
            for i in range(int(ng / 2))
        ]
        right_boundaries = [
            Boundary(xmax + (i + 1) * dx, btype.RIGHT, zeros(nv))
            for i in range(int(ng / 2))
        ]
        boundaries = left_boundaries + right_boundaries

        return Domain(interior, boundaries, components)

    def ilo(self):
        return self._nb

    def ihi(self):
        return self._nb + self._nx - 1

    def nb(self):
        return self._nb

    def cells(self):
        return self._domain

    def boundaries(self):
        return self._domain[: self._nb], self._domain[-self._nb :]

    def interior(self):
        return self._domain[self._nb : -self._nb]

    def set_interior(self, cells):
        self._nx = len(cells)
        self._domain = [*self._domain[: self._nb], *cells, *self._domain[-self._nb :]]

    def component_index(self, v: str):
        return self._components.index(v)

    def component_name(self, i: int):
        return self._components[i]

    def positions(self):
        return [cell.x() for cell in self.cells()]

    def values(self):
        value_list = []
        for cell in self.cells():
            value_list.append(cell.values())

        return value_list

    def update(self, dt: int, interior_residual_block: list[list[float]]):
        for i, cell in enumerate(self.interior()):
            cell.update(dt, interior_residual_block[i])

    def apply_BC(self, v: str, bc: str = "periodic", xmin=0.0, xmax=1.0):
        # Find index of component
        idx = self.component_index(v)

        if bc == "periodic":
            lb, rb = self.boundaries()

            # Get interior indices
            ilo = self.ilo()
            ihi = self.ihi()

            # left boundary
            # Ghost cells are the rightmost elements (same order)
            for i, b in enumerate(lb):
                b.set_x(xmin - (xmax - self._domain[ihi - (i + 1)].x()))
                b.set_value(idx, self._domain[ihi - (i + 1)].value(idx))

            # right boundary
            # Ghost cells are the leftmost elements (same order)
            for i, b in enumerate(rb):
                b.set_x(xmax + (self._domain[(i + 1) + ilo].x() - xmin))
                b.set_value(idx, self._domain[(i + 1) + ilo].value(idx))

        elif self.bc == "outflow":
            # Find the lowest and highest interior indices
            ilo = self._nb
            ihi = self._nx + self._nb - 1

            lb, rb = self.boundaries()
            # left boundary
            # Ghost cells are the leftmost interior element
            for i, b in enumerate(lb):
                b.set_value(idx, self._domain[ilo].value(idx))

            # right boundary
            # Ghost cells are the leftmost elements (same order)
            for i, b in enumerate(rb):
                b.set_value(idx, self._domain[ihi].value(idx))

        else:
            raise NotImplementedError
