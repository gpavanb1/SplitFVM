from .domain import Domain
from .constants import btype, btype_map
from .error import SFVM


def apply_BC(d: Domain, v: str, bc: dict = {"left": "periodic", "right": "periodic"}, xmin=0.0, xmax=1.0):
    """
    Apply boundary conditions to the given domain.

    Parameters
    ----------
    d : Domain
        The domain to apply the boundary conditions to.
    v : str
        The name of the variable to apply the boundary conditions to.
    bc : str, optional
        The type of boundary condition to apply. Acceptable values are "periodic" and "outflow". Default is "periodic".
    xmin : float, optional
        The minimum x-value of the domain. Default is 0.0.
    xmax : float, optional
        The maximum x-value of the domain. Default is 1.0.

    Raises
    ------
    NotImplementedError
        If an unsupported boundary condition is specified.
    """

    # Get cells
    cells = d.cells()

    # Find index of component
    idx = d.component_index(v)

    # Common values used in all types
    ilo = d.ilo()
    ihi = d.ihi()

    lb, rb = d.boundaries()

    # Check only required directions specified
    bc_keys = sorted(list(bc.keys()))
    if bc_keys != [btype_map[btype.LEFT], btype_map[btype.RIGHT]]:
        raise SFVM("Incorrect boundary directions specified")

    # Iterate over left and right BCs
    for dir in bc_keys:
        bc_type = bc[dir]

        if bc_type == "periodic":
            if dir == btype_map[btype.LEFT]:
                # left boundary
                # Ghost cells are (right to left) the rightmost elements (same order)
                for i, b in enumerate(lb):
                    shift = xmax - cells[ihi - (i + 1)].x()
                    b.set_x(xmin - shift)
                    b.set_value(idx, cells[ihi - (i + 1)].value(idx))

            elif dir == btype_map[btype.RIGHT]:
                # right boundary
                # Ghost cells are (left to right) the leftmost elements (same order)
                for i, b in enumerate(rb):
                    shift = cells[(i + 1) + ilo].x() - xmin
                    b.set_x(xmax + shift)
                    b.set_value(idx, cells[(i + 1) + ilo].value(idx))

            else:
                raise SFVM("Incorrect boundary direction encountered")

        elif bc_type == "outflow":
            if dir == btype_map[btype.LEFT]:
                # left boundary
                for i, b in enumerate(lb):
                    # The shift mirrors the interior on the same side
                    shift = cells[(i + 1) + ilo].x() - xmin
                    b.set_x(xmin - shift)
                    # Value same as leftmost interior
                    b.set_value(idx, cells[ilo].value(idx))

            elif dir == btype_map[btype.RIGHT]:
                # right boundary
                for i, b in enumerate(rb):
                    # The shift mirrors the interior on the same side
                    shift = xmax - cells[ihi - (i + 1)].x()
                    b.set_x(xmax + shift)
                    # Value same as extrapolated from interior
                    dy = cells[ihi].value(idx) - cells[ihi - 1].value(idx)
                    dx = cells[ihi].x() - cells[ihi - 1].x()
                    delta_x = cells[ihi + (i + 1)].x() - cells[ihi + i].x()

                    b.set_value(
                        idx, cells[ihi + i].value(idx) + (dy/dx) * delta_x)

            else:
                raise SFVM("Incorrect boundary direction encountered")

        # Dictionary-based boundary conditions
        # Dirichlet and Neumann BCs require additional values also
        elif isinstance(bc_type, dict) and len(bc_type.keys()) == 1:
            bc_data = list(bc_type.values())[0]
            bc_type = list(bc_type.keys())[0]

            if bc_type == "neumann":
                if dir == btype_map[btype.LEFT]:
                    # left boundary
                    neumann_value = bc_data
                    for i, b in enumerate(lb):
                        # The shift mirrors the interior on the same side
                        shift = cells[(i + 1) + ilo].x() - xmin
                        b.set_x(xmin - shift)

                        # Cell width to the left
                        dx = cells[ilo - i].x() - cells[ilo - (i + 1)].x()
                        b.set_value(
                            idx, cells[ilo - i].value(idx) - neumann_value * dx)

                elif dir == btype_map[btype.RIGHT]:
                    # right boundary
                    neumann_value = bc_data
                    for i, b in enumerate(rb):
                        # The shift mirrors the interior on the same side
                        shift = xmax - cells[ihi - (i + 1)].x()
                        b.set_x(xmax + shift)

                        # Cell width to the right
                        dx = cells[ihi + (i + 1)].x() - cells[ihi + i].x()
                        b.set_value(
                            idx, cells[ihi + i].value(idx) + neumann_value * dx)

                else:
                    raise SFVM("Incorrect boundary direction encountered")

            elif bc_type == "dirichlet":
                if dir == btype_map[btype.LEFT]:
                    # left boundary
                    dirichlet_value = bc_data
                    for i, b in enumerate(lb):
                        # The shift mirrors the interior on the same side
                        shift = cells[(i + 1) + ilo].x() - xmin
                        b.set_x(xmin - shift)

                        b.set_value(idx, dirichlet_value)

                elif dir == btype_map[btype.RIGHT]:
                    # right boundary
                    dirichlet_value = bc_data
                    for i, b in enumerate(rb):
                        # The shift mirrors the interior on the same side
                        shift = xmax - cells[ihi - (i + 1)].x()
                        b.set_x(xmax + shift)

                        b.set_value(idx, dirichlet_value)

                else:
                    raise SFVM("Incorrect boundary direction encountered")

            else:
                raise SFVM(
                    "Incorrect data specified for dictionary-type boundaries")
        else:
            raise SFVM("Boundary type not implemented")
