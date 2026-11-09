from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------

def _format_rad(val):
    frac = Fraction(val / np.pi).limit_denominator(100)
    num, den = frac.numerator, frac.denominator
    if num == 0:
        return "0"
    sign = "-" if num < 0 else ""
    abs_num = abs(num)
    num_str = "" if abs_num == 1 else f"{abs_num}"
    if den == 1:
        return f"${sign}{num_str}\\pi$"
    else:
        return f"${sign}\\frac{{{num_str}\\pi}}{{{den}}}$"


def _format_rad_ticks(orig_lim, unit_scale):
    vmin, vmax = np.sort(orig_lim) * unit_scale
    steps = np.array([1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 1, 2, 4]) * np.pi
    step = steps[np.argmin(np.abs((vmax - vmin) / steps - 4))]
    scaled_ticks = np.arange(np.ceil(vmin / step - 1e-9), np.floor(vmax / step + 1e-9) + 1) * step
    ticks = scaled_ticks / unit_scale
    ticklabels = [_format_rad(val) for val in scaled_ticks]
    return ticks, ticklabels

# --------------------------------------------------------------------------
# TICKS
# --------------------------------------------------------------------------

def format_axis_ticks(coord, ax, label="", unit="u.a.", unit_scale=1):
    coord = coord.lower()

    # Determine methods for ticks, axis labels, and limits
    if coord == "x":
        get_ticks, set_ticks = ax.get_xticks, ax.set_xticks
        set_ticklabels = ax.set_xticklabels
        set_label = ax.set_xlabel
        get_lim, set_lim = ax.get_xlim, ax.set_xlim

    elif coord == "y":
        get_ticks, set_ticks = ax.get_yticks, ax.set_yticks
        set_ticklabels = ax.set_yticklabels
        set_label = ax.set_ylabel
        get_lim, set_lim = ax.get_ylim, ax.set_ylim

    elif coord == "z":
        get_ticks, set_ticks = ax.get_zticks, ax.set_zticks
        set_ticklabels = ax.set_zticklabels
        set_label = ax.set_zlabel
        get_lim, set_lim = ax.get_zlim, ax.set_zlim

    else:
        raise ValueError(f"Unsupported coordinate: {coord}. Choose from 'x', 'y', 'z', or 'cbar'.")

    # Save current axis limits
    orig_lim = get_lim()

    ticks = get_ticks()
    scaled_ticks = ticks * unit_scale

    # Format tick labels
    if unit == "rad":
        ticks, ticklabels = _format_rad_ticks(orig_lim, unit_scale)
    else:
        ticklabels = [f"{val:g}" for val in scaled_ticks]

    set_ticks(ticks)
    set_ticklabels(ticklabels)
    set_lim(orig_lim)

    # Update axis/colorbar label with [unit]
    if unit is not None:
        set_label(f"{label} [{unit}]".strip())


def plot_spectrum(freqs, spectrum, peaks, ax=None, show=False, plot_harmonics=False):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(freqs, spectrum, color="k")
    ax.plot(freqs[peaks], spectrum[peaks], "x", color="red")

    if plot_harmonics:
        harmonic_positions = np.arange(np.min(freqs), np.max(freqs), freqs[peaks[0]])
        ax.vlines(harmonic_positions, ymin=0, ymax=np.max(spectrum), ls=":", color="lightgray", zorder=-10)

    if show:
        plt.show()
        return None

    else:
        return ax


def plot_video_fft(x, y, video_fft, loc, cbar_units="mm s$^{-1}$", cbar_unit_scale=1e3, axis_units="u.a", axis_unit_scale=1):
    X, Y = np.meshgrid(x, y)

    image = video_fft[loc]
    vmax = np.max(np.abs(image))

    fig, axs = plt.subplots(1, 4, figsize=(16, 3), sharey=True)

    c = axs[0].pcolormesh(X, Y, np.real(image), cmap="coolwarm", vmin=-vmax, vmax=vmax, rasterized=True)
    cbar = plt.colorbar(c)
    format_axis_ticks("y", label="Parte real", ax=cbar.ax, unit=cbar_units, unit_scale=cbar_unit_scale)

    c = axs[1].pcolormesh(X, Y, np.imag(image), cmap="coolwarm", vmin=-vmax, vmax=vmax, rasterized=True)
    cbar = plt.colorbar(c)
    format_axis_ticks("y", label="Parte imaginaria", ax=cbar.ax, unit=cbar_units, unit_scale=cbar_unit_scale)

    c = axs[2].pcolormesh(X, Y, np.abs(image), cmap="gray", vmin=0, vmax=vmax, rasterized=True)
    cbar = plt.colorbar(c)
    format_axis_ticks("y", label="Amplitud", ax=cbar.ax, unit=cbar_units, unit_scale=cbar_unit_scale)

    c = axs[3].pcolormesh(X, Y, np.angle(image), cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, rasterized=True)
    cbar = plt.colorbar(c)
    format_axis_ticks(coord="y", label="Fase", ax=cbar.ax, unit="rad")

    for ax in axs:
        format_axis_ticks(coord="x", label="$x$", ax=ax, unit=axis_units, unit_scale=axis_unit_scale)
        format_axis_ticks(coord="y", label="$y$", ax=ax, unit=axis_units, unit_scale=axis_unit_scale)

    return fig, axs
