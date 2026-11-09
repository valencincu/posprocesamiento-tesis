from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory

import h5py


def select_directory(initialdir=None):
    Tk().withdraw()
    directory = Path(askdirectory(title="Select directory", initialdir=initialdir))
    print(f"Analyzing: {directory}")
    return directory


def h5_tree(vals, pre=''):
    items = len(vals)
    for key, val in vals.items():
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
