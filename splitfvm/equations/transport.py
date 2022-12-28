from splitfvm.flux import fluxes


class TransportEquation:
    def __init__(self, F, D, S):
        self.F = F
        self.D = D
        self.S = S
        pass

    def residuals(self, cell_sub, scheme):
        # Cell width
        # Calculate for center cell
        # Average of distance between adjacent cell centers
        ic = len(cell_sub) // 2
        dxw = cell_sub[ic].x() - cell_sub[ic - 1].x()
        dxe = cell_sub[ic + 1].x() - cell_sub[ic].x()
        dx = 0.5 * (dxw + dxe)

        # Calculate fluxes
        Fw, Fe = fluxes(self.F, cell_sub, scheme)
        # DFw, DFe = diffusion_fluxes(self.D, cell_sub, scheme)
        rhs = self.S(cell_sub) - (1 / dx) * (Fw - Fe)  # + (DFe - DFw)
        return rhs
