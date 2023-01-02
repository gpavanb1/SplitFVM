from .domain import Domain


def apply_BC(d: Domain, v: str, bc: str = "periodic", xmin=0.0, xmax=1.0):
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

    if bc == "periodic":
        lb, rb = d.boundaries()

        # Get interior indices
        ilo = d.ilo()
        ihi = d.ihi()

        # left boundary
        # Ghost cells are the rightmost elements (same order)
        for i, b in enumerate(lb):
            b.set_x(xmin - (xmax - cells[ihi - (i + 1)].x()))
            b.set_value(idx, cells[ihi - (i + 1)].value(idx))

        # right boundary
        # Ghost cells are the leftmost elements (same order)
        for i, b in enumerate(rb):
            b.set_x(xmax + (cells[(i + 1) + ilo].x() - xmin))
            b.set_value(idx, cells[(i + 1) + ilo].value(idx))

    elif bc == "outflow":
        # Find the lowest and highest interior indices
        ilo = d.ilo()
        ihi = d.ihi()

        lb, rb = d.boundaries()
        # left boundary
        # Ghost cells are the leftmost interior element
        for i, b in enumerate(lb):
            b.set_x(xmin - (xmax - cells[ihi - (i + 1)].x()))
            b.set_value(idx, cells[ilo].value(idx))

        # right boundary
        # Ghost cells are the leftmost elements (same order)
        for i, b in enumerate(rb):
            b.set_x(xmax + (cells[(i + 1) + ilo].x() - xmin))
            b.set_value(idx, cells[ihi].value(idx))

    else:
        raise NotImplementedError
