# %% libraries
import h5py
import matplotlib.pyplot as plt
import numpy as np
import rlic
from scipy.signal import find_peaks

from postaux.ffts import (
    calculate_spectrum,
)
from postaux.file_management import h5_tree, select_directory
from postaux.visualization import format_axis_ticks, plot_spectrum, plot_video_fft
from postpiv.datasets import dataset_from_h5
from postpiv.trajectories import integrate_trajectories, seed_generator
from postpiv.visualization import animate

plt.style.use("science.mplstyle")

# -----------------------------------------------------------------------------
# %% CONSTANTS
# -----------------------------------------------------------------------------

START = 0
STOP = 500
PEAK_PROMINENCE = 2e-3

# -----------------------------------------------------------------------------
# %% SELECT DIRECTORY
# -----------------------------------------------------------------------------

directory = select_directory()
with h5py.File(directory / "piv.hdf5", "r") as f:
    cal_factor = f["calibration_factor"][()]
    x = f["x"][:]
    y = f["y"][:]
    xs, ys = np.meshgrid(x, y)
    us = f["u"][START:STOP]
    vs = f["v"][START:STOP]
    ts = f["t"][START:STOP]
    fps = 1 / np.mean(np.diff(ts))
    h5_tree(f)

# -----------------------------------------------------------------------------
# %% SPECTRAL ANALYSIS
# -----------------------------------------------------------------------------

freqs, ufft, uspectrum = calculate_spectrum(us, fps=fps)
upeaks, _ = find_peaks(uspectrum, prominence=PEAK_PROMINENCE)
ax=plot_spectrum(freqs, uspectrum, upeaks, plot_harmonics=True)
format_axis_ticks("x", label="Frecuencia", ax=ax, unit=r"s$^{-1}$")
format_axis_ticks("y", label="Amplitud", ax=ax, unit_scale=1e3, unit=r"mm")
plt.savefig(directory / "uspectrum.pdf")
plt.show()

freqs, vfft, vspectrum = calculate_spectrum(vs, fps=fps)
vpeaks, _ = find_peaks(vspectrum, prominence=PEAK_PROMINENCE)
ax=plot_spectrum(freqs, vspectrum, vpeaks, plot_harmonics=True)
format_axis_ticks("x", label="Frecuencia", ax=ax, unit=r"s$^{-1}$")
format_axis_ticks("y", label="Amplitud", ax=ax, unit_scale=1e3, unit=r"mm s$^{-1}$")
plt.savefig(directory / "vspectrum.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% VIDEO FFT; U
# -----------------------------------------------------------------------------

loc = 0
fig, axs = plot_video_fft(x, y, ufft, loc,
                             cbar_units="mm s$^{-1}$", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(fname=directory / f"u_loc{str(freqs[loc]).replace(".", ",")}.pdf")
plt.show()

loc = upeaks[0]
fig, axs = plot_video_fft(x, y, ufft, loc,
                             cbar_units="mm s$^{-1}$", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(directory / f"u_loc{str(freqs[upeaks[1]]).replace(".", ",")}.pdf")
plt.show()

loc = upeaks[1]
fig, axs = plot_video_fft(x, y, ufft, loc,
                             cbar_units="mm s$^{-1}$", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(fname=directory / f"u_loc{str(freqs[loc]).replace(".", ",")}.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% VIDEO FFT; V
# -----------------------------------------------------------------------------

loc = 0
fig, axs = plot_video_fft(x, y, vfft, loc,
                             cbar_units="mm s$^{-1}$", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(fname=directory / f"v_loc{str(freqs[loc]).replace(".", ",")}.pdf")
plt.show()

loc = vpeaks[0]
fig, axs = plot_video_fft(x, y, vfft, loc,
                             cbar_units="mm s$^{-1}$", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(fname=directory / f"v_loc{str(freqs[loc]).replace(".", ",")}.pdf")
plt.show()

loc = vpeaks[1]
fig, axs = plot_video_fft(x, y, vfft, loc,
                             cbar_units="mm s$^{-1}$", cbar_unit_scale=1e3, axis_units="cm", axis_unit_scale=1e2)

plt.savefig(fname=directory / f"v_loc{str(freqs[loc]).replace(".", ",")}.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% LINE INTEGRAL CONVOLUTION
# -----------------------------------------------------------------------------

prng = np.random.default_rng(0)
texture = prng.random(us[0].shape)
kernel = 1 - np.abs(np.linspace(-1, 1, 15))

fig, ax = plt.subplots(figsize=(4.9, 4.9))
image = rlic.convolve(
    texture,
    us[0],
    vs[0],
    kernel=kernel,
    boundaries="periodic",
    iterations=2,
)
ax.set_title("Line Integral Convolution \n (flujo estacionario)")
ax.pcolormesh(x, y, image, cmap="pink")
format_axis_ticks("x", ax, "$x$", unit="cm", unit_scale=1e2)
format_axis_ticks("y", ax, "$y$", unit="cm", unit_scale=1e2)
plt.savefig(directory / "lic.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% LAGRANGIAN TRAJECTORIES
# -----------------------------------------------------------------------------

initial_coordinates = seed_generator(
    300, ((np.min(x), np.max(x)), (np.min(y), np.max(y)))


)
lines = integrate_trajectories(
    ts,
    x,
    y,
    us,
    vs,
    initial_coordinates,
    n_steps=400
)

plt.figure(figsize=(4.9, 4.9))
for line in lines:
    plt.plot(line[:, 0], line[:, 1])

plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(y), np.max(y))
plt.title("Integración de trayectorias lagrangianas")
format_axis_ticks("x", plt.gca(), "$x$", unit="cm", unit_scale=1e2)
format_axis_ticks("y", plt.gca(), "$y$", unit="cm", unit_scale=1e2)
plt.savefig(directory / "lagrangian_trajectories.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% PIVPY: LOAD DATASET
# -----------------------------------------------------------------------------

ds = dataset_from_h5(directory, stop=500, chc=False);
ds.piv.vorticity()

# -----------------------------------------------------------------------------
# %% PIVPY: VORTICITY BEFORE FILTERS
# -----------------------------------------------------------------------------

fig = plt.figure(dpi=200)
ds.piv.showscal(cmap="coolwarm")
format_axis_ticks("x", plt.gca(), "$x$", unit="cm", unit_scale=1e2)
format_axis_ticks("y", plt.gca(), "$y$", unit="cm", unit_scale=1e2)
plt.gca().set_aspect("equal")
plt.show()

# -----------------------------------------------------------------------------
# PIVPY: PIV PLOT
# -----------------------------------------------------------------------------

fig, ax = ds.piv.plot(blur=0, quiver=False, streamline_color="k", streamline_density=1.5)
format_axis_ticks("x", ax, "$x$", unit="cm", unit_scale=1e2)
format_axis_ticks("y", ax, "$y$", unit="cm", unit_scale=1e2)
plt.savefig(directory / "piv_plot.pdf")
plt.show()

# -----------------------------------------------------------------------------
# %% PIVPY: ANIMATION
# -----------------------------------------------------------------------------

ds = dataset_from_h5(directory, stop=40, chc=None);
anim = animate(ds, interval=80)
anim.save(directory / "piv_animate.gif")
