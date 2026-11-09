import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as matplotlib_backend

from postaux.file_management import select_directory

matplotlib_backend("TkAgg")

# OPEN FILE
directory = select_directory()
f = h5py.File(directory / "height_maps.hdf5", "r")

# DATA
ts = f["t [s]"]
height_maps = f["height_maps [m]"]
shape = height_maps[0].shape

# PLAY VIDEO
plt.ion()

fig, ax = plt.subplots()
title = plt.suptitle("")
for i, frame in enumerate(height_maps):
    ax.clear()
    ax.imshow(frame, cmap="RdBu", vmin=-4e-3, vmax=4e-3)
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[0])
    title.set_text(f"{ts[i]*1000} ms")
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.close()

f.close()
