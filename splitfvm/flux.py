from enum import Enum
from .error import SFVM

# TODO: Add more schemes
Schemes = Enum("Schemes", "LF")
Directions = Enum("Directions", "WEST EAST")


def fluxes(F, cell_sub, scheme):
    if scheme == Schemes.LF:
        if len(cell_sub) != 3:
            raise SFVM("Improper stencil size for LF scheme")

        # West Flux
        ul = cell_sub[0].values()
        ur = cell_sub[1].values()
        Fl = F(ul)
        Fr = F(ur)
        u_diff = ur - ul
        # TODO: Evaluate spectral radius
        sigma = 0.1
        Fw = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

        # East Flux
        ul = cell_sub[1].values()
        ur = cell_sub[2].values()
        Fl = F(ul)
        Fr = F(ur)
        u_diff = ur - ul
        # TODO: Evaluate spectral radius
        sigma = 0.1
        Fe = 0.5 * (Fl + Fr) - 0.5 * sigma * u_diff

        return Fw, Fe
