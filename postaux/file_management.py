import numpy as np
import h5py

from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path
from pivpy import schema


def select_directory():
    Tk().withdraw() 
    directory = Path(askdirectory(title="Select data directory"))
    print(f"Analyzing: {directory}")
    return directory


def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


def dataset_from_h5py(directory, start=0, stop=10, xy_unit_scale=1, uv_unit_scale=1):
    with h5py.File(directory / 'piv.hdf5', 'r') as f:
        x = f["x [m]"][0,:] * xy_unit_scale
        y = f["y [m]"][:,0] * xy_unit_scale

        us = np.moveaxis(f["u [m per s]"][start:stop], 0, -1) * uv_unit_scale
        vs = np.moveaxis(f["v [m per s]"][start:stop], 0, -1) * uv_unit_scale

        ts = f["t [s]"][start:stop]

        ds = schema.build_dataset(x, y, us, vs, t=ts)

    return ds