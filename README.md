# Note

This repository has been archived. Please refer to [SplitFXM](https://github.com/gpavanb1/SplitFXM)

# SplitFVM

[![Downloads](https://pepy.tech/badge/splitfvm)](https://pepy.tech/project/splitfvm)

![img](https://github.com/gpavanb1/SplitFVM/blob/main/assets/logo.jpg)

1D [Finite-Volume](https://en.wikipedia.org/wiki/Finite_volume_method) with [adaptive mesh refinement](https://en.wikipedia.org/wiki/Adaptive_mesh_refinement) and steady-state solver using Newton and [Split-Newton](https://github.com/gpavanb1/SplitNewton) approach

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
d = Domain.from_size(20, 2, ["u", "v", "w"]) # nx, ng, variables

# Set IC and BC
ics = {"u": "gaussian", "v": "rarefaction"}
bcs = {
    "u": {
        "left": "periodic",
        "right": "periodic"
    },
    "v": {
        "left": {"dirichlet": 3},
        "right": {"dirichlet": 4}
    },
    "w": {
        "left": {"dirichlet": 2},
        "right": "periodic"
    }
}
s = Simulation(d, m, ics, bcs)

# Advance in time or to steady state
s.evolve(dt=0.1)
bounds = [[-1., -2., 0.], [5., 4., 3.]]
iter = s.steady_state(split=True, split_loc=1, bounds=bounds)

# Visualize
draw(d, "label")
```

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1)
for any questions.

## Acknowledgements

Do visit its [Finite-Difference](https://github.com/gpavanb1/SplitFDM) cousin

Special thanks to [Cantera](https://github.com/Cantera/cantera) and [WENO-Scalar](https://github.com/comp-physics/WENO-scalar) for serving as an inspiration for code architecture
