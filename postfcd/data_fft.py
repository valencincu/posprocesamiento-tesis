import numpy as np
import h5py
import os
import scienceplots
from uncertainties import ufloat
from matplotlib import pyplot as plt
from matplotlib import use as matplotlib_backend
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from tkinter import Tk
from tkinter.filedialog import askdirectory
from uncertainties import unumpy as unp
import scipy.fft as fft

matplotlib_backend("TkAgg")
plt.style.use('science')
cm = 1/2.54

ffts  = []
freqs = []
colors = plt.cm.hsv(np.linspace(0,1,8))

def video_fft(frames, fps=1):
    frames_fft = fft.rfft(frames, axis=0, norm='forward')
    freqs = fft.rfftfreq(len(frames), 1/fps)
    return freqs, frames_fft

def video_ifft(freqs, frames_fft, freq_loc=None, sigmas=1):
    if freq_loc is not None:
        delta = (freqs[1] - freqs[0])*sigmas
        frames_fft = frames_fft.copy()
        frames_fft[np.abs(freqs - freq_loc) > delta,...] = 0

    video_ifft = fft.irfft(frames_fft, axis=0)
    return video_ifft

# %% OPEN FILE
Tk().withdraw() 
directory = Path(askdirectory(title="Select data directory"))
print(f"Analyzing: {directory}")
f = h5py.File(directory / 'height_maps.hdf5', 'r')

# %% DATA
t = f["t [s]"]
x = f["x [m]"][:,:]
y = f["y [m]"][:,:]
height_maps = f['height_maps [m]']

shape = height_maps.shape
fps = 1 / np.mean(np.diff(t))

# %% NEAR FIELD
heights  = height_maps[:,:,:]
heights -= heights.mean()

# %% FREQUENCY ANALYSIS
freqs, fft = video_fft(heights, fps=fps)
peaks, _   = find_peaks(np.max(np.abs(fft), axis=(1,2)), distance=1, prominence=0.5e-7, height=0.1*np.max(np.abs(fft)))
if peaks.size == 0:
	raise ValueError("No frequency peaks found in the height-map time series.")

peak_freq = ufloat(freqs[peaks[0]], np.mean(np.diff(freqs)))
print("Peak frequiency:", peak_freq)


# %% SPECTRAL PLOT
fig = plt.figure(figsize=(8*cm, 4*cm))
fig.tight_layout()

plt.plot(freqs, np.max(np.abs(fft)*1e3, axis=(1,2)), color='k', lw=1)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"$\text{max}(\text{FFT})$ [mm]")
plt.savefig(directory / "espectro.pdf", dpi=200)
plt.show()

# %% FILTERED PLOT
fig, axs = plt.subplots(2, 1, figsize=(8*cm, 13*cm), sharex=True)
fig.tight_layout()

im = axs[0].pcolormesh(x*1e3, y*1e3, np.abs(fft[peaks[0]])*1e3, cmap='bone', rasterized=True)
plt.colorbar(im, label=r"Amplitud [mm]", ax=axs[0])

axs[0].set_ylabel("$y$ [mm]")
axs[0].set_aspect("equal")

im = axs[1].pcolormesh(x*1e3, y*1e3, np.angle(fft[peaks[0]]), cmap='hsv', rasterized=True)
plt.colorbar(im, label="Fase [rad]", ax=axs[1])

axs[1].set_xlabel("$x$ [mm]")
axs[1].set_ylabel("$y$ [mm]")
axs[1].set_aspect("equal")

plt.savefig(directory / "fcd_fft.pdf", dpi=200)
plt.show()
f.close()