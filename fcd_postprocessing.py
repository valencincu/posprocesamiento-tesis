# %% libraries
# -----------------------------------------------------------------------------

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from postaux.ffts import (
    calculate_spectrum,
    find_wavenumber_peaks,
    image_fft,
    wavenumber_peak_alignment,
)
from postaux.file_management import h5_tree, select_directory
from postaux.visualization import format_axis_ticks, plot_spectrum, plot_video_fft

plt.style.use("science.mplstyle")

# -----------------------------------------------------------------------------
# %% CONSTANTS
# -----------------------------------------------------------------------------

START = 0
STOP = 500
PEAK_PROMINENCE = 1e-5

# -----------------------------------------------------------------------------
# %% SELECT DIRECTORY
# -----------------------------------------------------------------------------

directory = select_directory(initialdir="data/fcd")
with h5py.File(directory / "height_maps.hdf5", "r") as f:
    cal_factor = f.attrs["calibration_factor"]

    x = f["x"][:]
    y = f["y"][:]
    xs, ys = np.meshgrid(x, y)
    height_maps = f["height_maps"][START:STOP]
    ts = f["t"][START:STOP]
    fps = 1 / np.mean(np.diff(ts))
    h5_tree(f)

# -----------------------------------------------------------------------------
# %% SPECTRAL ANALYSIS
# -----------------------------------------------------------------------------

freqs, fft, spectrum = calculate_spectrum(height_maps, fps=fps)
peaks, _ = find_peaks(spectrum, prominence=PEAK_PROMINENCE)
ax = plot_spectrum(freqs, spectrum, peaks, plot_harmonics=True)
format_axis_ticks("x", label="Frecuencia", ax=ax, unit=r"s$^{-1}$")
format_axis_ticks("y", label="Amplitud", ax=ax, unit_scale=1e3, unit=r"mm")
plt.savefig(directory / "spectrum.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% VIDEO FFT
# -----------------------------------------------------------------------------

loc = 0
fig, axs = plot_video_fft(x, y, fft, loc,
                             cbar_units="mm", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(fname=directory / "loc0.pdf")
plt.show()

loc = peaks[0]
fig, axs = plot_video_fft(x, y, fft, loc,
                             cbar_units="mm", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(directory / f"loc{str(freqs[peaks[0]]).replace(".", ",")}.pdf")
plt.show()

loc = peaks[1]
fig, axs = plot_video_fft(x, y, fft, loc,
                             cbar_units="mm", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(directory / f"loc{str(freqs[peaks[1]]).replace(".", ",")}.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% STUDY OF WAVEVECTORS
# -----------------------------------------------------------------------------

k2_fft = image_fft(fft[peaks[0]])
plt.imshow(np.abs(k2_fft), zorder=-10)

blobs = find_wavenumber_peaks(np.abs(k2_fft), threshold=4)
alignment = wavenumber_peak_alignment(blobs[0], blobs[1], k2_fft.shape)
print(alignment)

blobs=np.array(blobs).T
plt.scatter(blobs[1], blobs[0], color="red")
plt.xlim(380, 410)
plt.ylim(380, top=410)
plt.show()
