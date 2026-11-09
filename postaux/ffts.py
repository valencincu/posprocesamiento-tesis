import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, irfft, rfft, rfftfreq
from scipy.stats import mode
from skimage.feature import peak_local_max

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

PI = np.pi

# --------------------------------------------------------------------------
# COMMON FUNCTIONS
# --------------------------------------------------------------------------

def wavenumber_axis(size, calibration_factor=1, shifted=True):
    frequencies = fftfreq(size, calibration_factor / (2.0 * np.pi))
    return fftshift(frequencies) if shifted else frequencies

# --------------------------------------------------------------------------
# VIDEO UTILITIES
# --------------------------------------------------------------------------

def video_fft(frames, fps=1):
    frames_fft = rfft(frames, axis=0, norm='forward')
    freqs = rfftfreq(len(frames), 1/fps)
    return freqs, frames_fft


def video_ifft(freqs, frames_fft, freq_loc=None, sigmas=1):
    if freq_loc is not None:
        delta = (freqs[1] - freqs[0])*sigmas
        frames_fft = frames_fft.copy()
        frames_fft[np.abs(freqs - freq_loc) > delta,...] = 0

    video_ifft = irfft(frames_fft, axis=0)
    return video_ifft

# --------------------------------------------------------------------------
# 2D FFT UTILITIES
# --------------------------------------------------------------------------

def pixels_to_wavenumbers(image_shape, locations, calibration_factor=1):
    k_space_rows = wavenumber_axis(image_shape[0], calibration_factor, shifted=True)
    k_space_cols = wavenumber_axis(image_shape[1], calibration_factor, shifted=True)

    if isinstance(locations[0], np.ndarray):
        wavenumbers =  np.array([[k_space_rows[loc[0]], k_space_cols[loc[1]]] for loc in locations])
    else:
        wavenumbers = np.array([k_space_rows[locations[0]], k_space_cols[locations[1]]])

    return wavenumbers


def image_fft(image):
    return fftshift(fft2(image - np.mean(image)))


def find_wavenumber_peaks(image_fft, threshold=1e-3):

    magnitude = np.abs(image_fft)
    coords = peak_local_max(
        magnitude,
        min_distance=1,
        threshold_abs=threshold,
        exclude_border=False,
    )

    # Sort peaks by intensity in descending order to match original behavior
    intensities = magnitude[tuple(coords.T)]
    sorted_indices = np.argsort(intensities)[::-1]

    return [coords[i] for i in sorted_indices]


def wavenumber_peak_alignment(peak_a, peak_b, image_shape):
    peak_a_frequency   = pixels_to_wavenumbers(image_shape, peak_a)
    peak_b_frequency = pixels_to_wavenumbers(image_shape, peak_b)
    return np.abs(np.dot(peak_a_frequency, peak_b_frequency)) / (np.hypot(*peak_a_frequency) * np.hypot(*peak_b_frequency))


def calculate_phase_mode(image, res=5, amin=-135, amax=135):
    angles = np.rad2deg(np.angle(image))
    selected_angles = angles[np.logical_and(angles < amax, angles > amin)] // res * res
    phase_mode = mode(selected_angles)

    return np.deg2rad(phase_mode.mode), phase_mode.count


def calculate_phase_mean(image, amin=-np.pi, amax=np.pi):
    angles = np.angle(image)
    angles = angles[np.logical_and(angles >= amin, angles <=amax)]
    print(angles)
    return np.mean(angles)


def calculate_spectrum(frames, fps, show=False):
    freqs, frames_fft = video_fft(frames, fps=fps)
    spectrum = np.max(np.abs(frames_fft), axis=(1,2))
    return freqs, frames_fft, spectrum


if __name__ == "__main__":
    peak_a = np.array([10, 10])
    peak_b = np.array([5, 0])

    print(wavenumber_peak_alignment(peak_a, peak_b, (11, 11)))
    print(np.sin(np.pi/4))

    z1 = 1/2 + 1j
    z2 = 1j
    z3 = 1j + 1/2
    z4 = 1j
