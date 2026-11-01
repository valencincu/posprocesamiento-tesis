# %% LIBRARIES
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from lick import lick_box_plot

from postaux.file_management import h5_tree, select_directory
from postpiv.datasets import dataset_from_h5py
from postpiv.visualization import animate

plt.style.use("science")

# %% SELECT DIRECTORY
directory = select_directory()
with h5py.File(directory / 'piv.hdf5', 'r') as f:
    h5_tree(f)

# %% LOAD DATASET
ds = dataset_from_h5py(directory, stop=50)

# %% BEFORE FILTERS
ds.piv.showscal(cmap="RdBu_r")

# %% PIV PLOT
ds.piv.plot(blur=0.0, quiver=False, streamline_color="k", streamline_density=1.5);

# %% ANIM
anim = animate(ds, interval=80)
anim.save(directory / "piv_animate.gif")

# %% LINE INTEGRAL CONVOLUTION
fig, ax = plt.subplots(figsize=(10,10))
lick_box_plot(
    fig,
    ax,
    ds["x"].values,
    ds["y"].values,
    ds["u"][:,:,0].values,
    ds["v"][:,:,0].values,
    ds["u"][:,:,0].values**2 + ds["v"][:,:,0].values**2,
    size_interpolated=256,
    niter_lic=5,
    post_lic="north-west-light-source",
    cmap="inferno",
    stream_density=0.5,
)
plt.show()
