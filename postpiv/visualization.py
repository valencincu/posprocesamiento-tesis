    
import numpy as np
import matplotlib.pyplot as plt

def piv_plot(x, y, u, v, flags=None, background=None, ax=None, scale=None, show=False):
    mask = np.zeros_like(u) if flags is None else flags
    if ax is None:
        _, ax = plt.subplots(dpi=100)
    if background is not None:
        ax.pcolormesh(background, cmap="gray", rasterized=True)
    
    ax.quiver(x, y, u * (1 - mask), v * (1 - mask), scale=scale)
    ax.set_aspect('equal')
    
    if show: plt.show()
    return ax