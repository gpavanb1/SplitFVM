from matplotlib.pyplot import plot
from .domain import Domain


def draw(d: Domain, l: str):
    """
    Plot the values of the domain.

    Parameters
    ----------
    d : Domain
        The domain to plot values for.
    l : str
        The label to use for the plot.
    """

    plot(d.positions(), d.values(), "-o", label=l)
