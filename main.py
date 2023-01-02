from splitfvm.domain import Domain
from splitfvm.simulation import Simulation
from splitfvm.visualize import draw
from matplotlib.pyplot import legend, show

from examples.advection_diffusion import AdvectionDiffusion

import argparse
import logging
from copy import deepcopy

# Set logging level
parser = argparse.ArgumentParser()
parser.add_argument(
    "--log",
    dest="loglevel",
    help="Set the loglevel for your solver  (DEBUG, INFO, WARNING, CRITICAL, ERROR)",
    type=str,
    default="WARNING",
)
args = parser.parse_args()
loglevel = getattr(logging, args.loglevel.upper())
logging.basicConfig(level=loglevel)

# Define the problem
m = AdvectionDiffusion(c=0.2, nu=0.001)
d = Domain.from_size(20, 2, ["u", "v"])
ics = {"u": "gaussian", "v": "rarefaction"}
bcs = {"u": "periodic", "v": "periodic"}
s = Simulation(d, m, ics, bcs)

# Initial domain
d_init = deepcopy(d)

# Advance in time
s.evolve(0.01)
iter = s.steady_state()
print(f"Took {iter} iterations")

# Show plot
draw(d_init, "l1")
draw(d, "l2")
legend()
show()
