import numpy as np

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

    def num_components(self):
        return len(self._components)

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

    def listify_interior(self, split, split_loc):
        interior_values = self.values()[self._nb : -self._nb]

        if not split:
            return np.array(interior_values).flatten()
        else:
            if split_loc is None:
                raise SFVM("Split location must be specified in this case")

            num_points = len(interior_values)
            ret = []
            # First add all the outer-block values
            for i in range(num_points):
                ret.extend(interior_values[i][:split_loc])
            # Then add all the inner block values
            for i in range(num_points):
                ret.extend(interior_values[i][split_loc:])

            return np.array(ret)

    def update(self, dt: int, interior_residual_block: list[list[float]]):
        for i, cell in enumerate(self.interior()):
            cell.update(dt, interior_residual_block[i])
