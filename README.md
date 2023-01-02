# SplitFVM

[![Downloads](https://pepy.tech/badge/splitfvm)](https://pepy.tech/project/splitfvm)

![img](assets/logo.png)

1D Finite-Volume with adaptive mesh refinement and steady-state solver using Newton and Split-Newton approach

## What does 'split' mean?

The system is divided into two and for ease of communication, let's refer to first set of variables as "outer" and the second as "inner".

* Holding the outer variables fixed, Newton iteration is performed till convergence using the sub-Jacobian

* One Newton step is performed for the outer variables with inner held fixed (using its sub-Jacobian)

* This process is repeated till convergence criterion is met for the full system (same as in Newton)

## How to install and execute?

Just run 
```
pip install splitfvm
```

There is an [examples](https://github.com/gpavanb1/SplitFVM/examples) folder that contains a test model - [Advection-Diffusion](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation)

You can define your own equations by simply creating a derived class from `Model` and adding to the `_equations` using existing or custom equations!

A basic driver program is as follows
```
# Define the problem
m = AdvectionDiffusion(c=0.2, nu=0.001)

# Define the domain and variables
# ng stands for ghost cell count
d = Domain.from_size(nx=20, ng=2, ["u", "v"])

# Set IC and BC
ics = {"u": "gaussian", "v": "rarefaction"}
bcs = {"u": "periodic", "v": "periodic"}
s = Simulation(d, m, ics, bcs)

# Advance in time or to steady state
s.evolve(dt=0.1)
iter = s.steady_state()
```

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1)
for any questions.