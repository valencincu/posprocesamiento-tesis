import h5py
import numpy as np
import xarray as xr
from pivpy import schema

POS_UNITS = "m"
TIME_UNITS = "s"
VEL_UNITS = r"m s$^{-1}$"

DELTA_T: float = 0.0

STANDARD_NAMES = {
    "u": "u_velocity",
    "v": "v_velocity",
    "chc": "validity_flag",
}

SCHEMA_VERSION = "1"


def _set_default_units(dataset: xr.Dataset) -> xr.Dataset:
    """Apply default units/standard_name/global attrs (sets missing only)."""
    ds = dataset
    ds.attrs["spatial_units"] = POS_UNITS
    ds["x"].attrs["units"] = POS_UNITS
    ds["y"].attrs["units"] = POS_UNITS
    ds["t"].attrs["units"] = TIME_UNITS
    for var in ("u", "v"):
        ds[var].attrs["units"] = VEL_UNITS
        ds[var].attrs.setdefault("standard_name", STANDARD_NAMES[var])

    ds.attrs["delta_t"] = float(ds["t"][1] - ds["t"][0])
    return ds

def dataset_from_h5py(directory, start=0, stop=10, xy_unit_scale=1, uv_unit_scale=1):
    with h5py.File(directory / 'piv.hdf5', 'r') as f:
        x = f["x [m]"][0,:] * xy_unit_scale
        y = f["y [m]"][:,0] * xy_unit_scale

        us = np.moveaxis(f["u [m per s]"][start:stop], 0, -1) * uv_unit_scale
        vs = np.moveaxis(f["v [m per s]"][start:stop], 0, -1) * uv_unit_scale

        ts = f["t [s]"][start:stop]

        ds = schema.build_dataset(x, y, us, vs, t=ts, )

    return _set_default_units(ds)
