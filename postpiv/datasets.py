import h5py
import numpy as np
import xarray as xr
from pivpy import schema
from xarray.core.dataset import Dataset

DELTA_T: float = 0.0

STANDARD_NAMES = {
    "u": "u",
    "v": "v",
    "chc": "validity_flag",
}

SCHEMA_VERSION = "1"

def _set_dataset_units(dataset: xr.Dataset, spatial_units: str = "m", time_units: str = "s") -> xr.Dataset:
    """Apply default units/standard_name/global attrs (sets missing only)."""
    ds: Dataset = dataset

    ds.attrs["spatial_units"] = spatial_units
    ds["x"].attrs["units"] = spatial_units
    ds["y"].attrs["units"] = spatial_units
    ds["t"].attrs["units"] = time_units
    for var in ("u", "v"):
        ds[var].attrs["units"] = f"{spatial_units}/{time_units}"
        ds[var].attrs.setdefault("standard_name", STANDARD_NAMES[var])

    if "chc" in ds:
        ds["chc"].attrs.setdefault("standard_name", STANDARD_NAMES["chc"])

    ds.attrs["delta_t"] = float(ds["t"][1] - ds["t"][0])
    return ds

def dataset_from_h5(directory, start=0, stop=10, chc=True):
    with h5py.File(directory / 'piv.hdf5', 'r') as f:
        x = f["x"]
        y = f["y"]

        us = np.moveaxis(f["u"][start:stop], 0, -1)
        vs = np.moveaxis(f["v"][start:stop], 0, -1)
        ts = f["t"][start:stop]

        if chc:
            chc = 1 - np.moveaxis(f["flags"][start:stop], 0, -1)
        else:
            chc = None

        ds = schema.build_dataset(x, y, us, vs, t=ts, chc=chc)
        return _set_dataset_units(ds, f.attrs["spatial_units"], f.attrs["time_units"])


def dataset_from_arrays(ts, x, y, us, vs, chc=None):
    us = np.moveaxis(us, 0, -1)
    vs = np.moveaxis(vs, 0, -1)
    ds = schema.build_dataset(x, y, us, vs, t=ts, chc=chc)
    return _set_dataset_units(ds)
