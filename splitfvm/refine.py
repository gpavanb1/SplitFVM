import math
from .cell import Cell
from .domain import Domain
from .error import SFVM


class Refiner:
    def __init__(self):
        # Default values
        # Borrowed from Cantera
        self._slope = 0.8
        self._curve = 0.8
        # Negative prune factor disables it
        self._prune = -0.1

        # Maximum points in grid
        self._npmax = 1000

        # Minimum range span factor
        self._min_range = 0.01

        # Minimum grid spacing
        self._min_grid = 1e-10

    def set_criteria(self, slope, curve, prune):
        self._slope = slope
        self._curve = curve
        self._prune = prune

    def set_max_points(self, npmax):
        self._npmax = npmax

    @classmethod
    # https://cantera.org/documentation/docs-2.5/doxygen/html/dd/d3c/refine_8cpp_source.html
    # Using only slope, curve and prune
    def refine(cls, d: Domain):
        cells = d.interior()
        n = len(cells)

        # Keep map
        # 1 means cell stays and -1 means it goes
        # Loc map
        # 1 means add a point there
        # c map
        # Addition due to that variable
        keep = {}
        c = {}
        loc = {}

        if len(cells) > cls._npmax:
            raise SFVM("Exceeded maximum number of points")

        dz = [cells[i + 1].x() - cells[i].x() for i in range(n - 1)]
        # nv -> Number of variables
        nv = len(cells[1].values())

        for i in range(nv):
            name = cls._components[i]
            # Slopes (s) for component i
            s = [
                (cells[j + 1].value(i) - cells[j].value(i))
                / (cells[j + 1].x() - cells[j].x())
                for j in range(n - 1)
            ]

            # Range of slopes
            smin = min(s)
            smax = max(s)

            # Max absolute values
            ss = max(abs(smin), abs(smax))

            # refine based on the slope of component i only if the
            # range of s is greater than a fraction 'min_range' of max
            # |s|. This eliminates components that consist of small
            # fluctuations on a constant slope background.
            if (smax - smin) > cls._min_range * ss:
                # maximum allowable difference in slope between
                # adjacent points
                dmax = cls._curve * (smax - smin)
                for j in range(n - 2):
                    r = abs(s[j + 1] - s[j]) / (dmax + math.ulp / dz[j])
                    if (
                        r > 1.0
                        and dz[j] >= 2 * cls._min_grid
                        and dz[j + 1] >= 2 * cls._min_grid
                    ):
                        c[name] = 1
                        loc[j] = 1
                        loc[j + 1] = 1

                    if r >= cls._prune:
                        keep[j + 1] = 1
                    elif keep[j + 1] == 0:
                        keep[j + 1] = -1

        # Don't allow pruning to remove multiple adjacent grid points
        # in a single pass.
        for j in range(2, n - 1):
            if keep[j] == -1 and keep[j - 1] == -1:
                keep[j] = 1

        cls.show_changes(loc, c)

        #######
        # AMR
        # Need to mark for deletion before deleting
        # as cell addition indices need to make sense
        #######
        # Iterate over m_keep and remove points
        for i, cell in enumerate(cells):
            if keep[i] == -1:
                cell.to_delete = True

        # Add cells at loc
        for i in loc.keys():
            x = 0.5 * (cells[i + 1].x() + cells[i].x())
            value = 0.5 * (cells[i + 1].values() + cells[i].values())
            cells.insert(i, Cell(x, value))

        # Delete marked cells
        for cell in cells:
            if cell.to_delete:
                del cell

    @classmethod
    def show_changes(cls, loc, c):
        if len(loc) != 0:
            print("#" * 78)
            print("Refining grid...")
            print("New points inserted after grid points ")
            for i in loc.keys():
                print(i, end=" ")
            print("    to resolve ", end="")
            for name in c.keys():
                print(name, end=" ")
            print("")
            print("#" * 78)
        else:
            print("No new points needed")
