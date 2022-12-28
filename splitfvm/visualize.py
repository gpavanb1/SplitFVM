from matplotlib.pyplot import plot
from .domain import Domain


def draw(d: Domain, l: str):
    plot(d.positions(), d.values(), "-o", label=l)
