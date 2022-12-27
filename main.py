import matplotlib.pyplot as plt
from models.advection import Advection
from splitfvm.domain import Domain
from splitfvm.simulation import Simulation
from splitfvm.initialize import set_initial_condition
from splitfvm.visualize import draw


m = Advection(c=0.1)
d = Domain.from_size(20, 2, ["u"])
set_initial_condition(d, "u")
s = Simulation(d, m)

# Initial domain
d.apply_BC("u")
draw(d, "l1")

# Advance in time
for i in range(10):
    s.evolve(0.02)
draw(d, "l2")

# Show plot
plt.legend()
plt.show()
