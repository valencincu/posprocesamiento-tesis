# %% LIBRARIES
from postaux.file_management import dataset_from_h5py, h5_tree, select_directory

# %% SELECT DIRECTORY
directory = select_directory()
h5_tree(directory)
ds = dataset_from_h5py(directory, stop=50)

# %% STREAMSCAL
ds.piv.streamscal();

# %% PIV PLOT
ds.piv.plot();


# %% ANIM
anim = ds.piv.animate(interval=80)

anim.save(directory / "piv_animate.gif")
