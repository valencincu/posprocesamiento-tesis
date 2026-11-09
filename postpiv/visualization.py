import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes


def piv_plot(x, y, u, v, flags=None, background=None, ax=None, scale=None, show=False):
    mask = np.zeros_like(u) if flags is None else flags
    if ax is None:
        _, ax = plt.subplots(dpi=100)
    if background is not None:
        ax.pcolormesh(background, cmap="gray", rasterized=True)

    ax.quiver(x, y, u * (1 - mask), v * (1 - mask), scale=scale)
    ax.set_aspect('equal')

    if show: plt.show()
    return ax


def animate(
    data: xr.Dataset,
    *,
    background: str | bool | None = "vorticity",
    quiver: bool = True,
    blur: float = 1.5,
    skip: int | None = None,
    arrow_scale: float | None = None,
    arrow_width: float = 0.0065,
    arrow_color: str = "#1a1a1a",
    arrow_alpha: float = 0.75,
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    interval: int = 80,
    repeat: bool = True,
    blit: bool = False,
    ax: Axes | None = None,
    title_fmt: str = "Flow Field (t = {t:.2f})",
    image: np.ndarray | None = None,
    image_extent: tuple[float, float, float, float] | None = None,
    image_alpha: float = 0.6,
    image_cmap: str = "gray",
    color_by: str | None = None,
    **kwargs,
):
    """Create a high-performance, beautiful Matplotlib FuncAnimation for time-series flow fields.

    Dynamically updates both background scalar fields (e.g. vorticity contours) and
    quiver velocity vectors in-place (``quiver.set_UVC``) for smooth, high-fps animation.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with time dimension 't' and velocity components ('u', 'v').
    background : str, bool, or None, default "vorticity"
        Background scalar field ('vorticity', 'mag', 'ke', 'divergence', or variable name).
    quiver : bool, default True
        Whether to draw velocity vectors.
    blur : float, default 1.5
        Gaussian filter smoothing sigma for background.
    skip : int or (skip_rows, skip_cols), optional
        Arrow subsampling step, or an independent (skip_rows, skip_cols)
        pair (see plot()). Default ~16 arrows per axis.
    arrow_scale : float, optional
        Arrow scaling factor.
    arrow_width : float, default 0.0065
        Shaft width factor for arrows.
    arrow_color : str, default "#1a1a1a"
        Arrow color.
    arrow_alpha : float, default 0.75
        Arrow opacity.
    cmap : str, optional
        Colormap for background scalar.
    clim : tuple[float, float], optional
        Global colorbar limits (vmin, vmax). If None, calculated robustly across all frames.
    interval : int, default 80
        Delay between frames in milliseconds (~12.5 fps).
    repeat : bool, default True
        Whether the animation loops.
    blit : bool, default False
        Whether blitting is used.
    ax : Axes, optional
        Existing axes to draw on.
    title_fmt : str, default "Flow Field (t = {t:.2f})"
        Format string for dynamic frame title.
    image : numpy.ndarray, optional
        Static raw camera frame drawn via imshow (same for every frame) when
        background="image", PIVlab-style.
    image_extent, image_alpha, image_cmap :
        As in plot(), for the background="image" frame.
    color_by : str, optional
        Color the quiver arrows per-frame by this field ("mag"/"speed", or a
        variable name in `data`) instead of a flat `arrow_color`.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (viewable in Jupyter/Marimo or saved via ``anim.save('flow.gif')``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.ndimage import gaussian_filter

    n_frames = int(data.sizes.get("t", 1))

    if ax is None:
        fig, target_ax = plt.subplots(figsize=(6.5, 5.2), dpi=120)
    else:
        target_ax = ax
        fig = ax.figure

    x_arr = np.asarray(data["x"].values, dtype=float)
    y_arr = np.asarray(data["y"].values, dtype=float)
    x2d, y2d = np.meshgrid(x_arr, y_arr)

    # Pre-extract or compute scalar background for all frames
    bg_str = str(background).lower() if (background is not None and not isinstance(background, bool)) else ("vorticity" if background is True or background == "vorticity" else None)
    has_bg = bg_str is not None and bg_str not in {"none", "false", "off"}

    if bg_str == "image" and image is not None:
        extent = image_extent if image_extent is not None else (
            float(x_arr.min()), float(x_arr.max()), float(y_arr.min()), float(y_arr.max())
        )
        target_ax.imshow(image, cmap=image_cmap, extent=extent, origin="upper",
                          alpha=float(image_alpha), zorder=1)
        has_bg = False  # static image, not a per-frame pcolormesh

    bg_frames = []
    use_cmap = cmap
    if has_bg:
        dx = abs(float(x_arr[1] - x_arr[0])) if len(x_arr) > 1 else 1.0
        dy = abs(float(y_arr[1] - y_arr[0])) if len(y_arr) > 1 else 1.0
        for fi in range(n_frames):
            ds_fi = data.isel(t=fi) if "t" in data.dims else data
            u_fi = np.asarray(ds_fi["u"].values, dtype=float)
            v_fi = np.asarray(ds_fi["v"].values, dtype=float)

            if bg_str in {"vorticity", "curl", "w"}:
                if "w" in ds_fi:
                    sc = np.asarray(ds_fi["w"].values, dtype=float)
                else:
                    dudx, dudy = np.gradient(u_fi, dx, dy, axis=(1, 0))
                    dvdx, dvdy = np.gradient(v_fi, dx, dy, axis=(1, 0))
                    sc = dvdx - dudy
                if use_cmap is None:
                    use_cmap = "RdBu_r"
            elif bg_str in {"mag", "speed"}:
                sc = np.sqrt(u_fi**2 + v_fi**2)
                if use_cmap is None:
                    use_cmap = "viridis"
            elif bg_str in {"ke", "energy"}:
                sc = 0.5 * (u_fi**2 + v_fi**2)
                if use_cmap is None:
                    use_cmap = "plasma"
            elif bg_str in {"divergence", "div"}:
                dudx, _ = np.gradient(u_fi, dx, axis=1)
                _, dvdy = np.gradient(v_fi, dy, axis=0)
                sc = dudx + dvdy
                if use_cmap is None:
                    use_cmap = "RdBu_r"
            elif bg_str in ds_fi:
                sc = np.asarray(ds_fi[bg_str].values, dtype=float)
                if use_cmap is None:
                    use_cmap = "viridis"
            else:
                sc = np.zeros_like(u_fi)
                if use_cmap is None:
                    use_cmap = "viridis"

            if blur and float(blur) > 0.0:
                sc = gaussian_filter(sc, sigma=float(blur))
            bg_frames.append(sc)

        if clim is not None:
            vmin, vmax = float(clim[0]), float(clim[1])
        elif use_cmap == "RdBu_r":
            all_finite = np.concatenate([f[np.isfinite(f)] for f in bg_frames if f.size])
            v_max = float(np.nanpercentile(np.abs(all_finite), 99)) if all_finite.size else 1.0
            v_max = max(v_max, 1e-9)
            vmin, vmax = -v_max, v_max
        else:
            all_finite = np.concatenate([f[np.isfinite(f)] for f in bg_frames if f.size])
            vmin = float(np.nanmin(all_finite)) if all_finite.size else 0.0
            vmax = float(np.nanpercentile(all_finite, 99)) if all_finite.size else 1.0

        bg_mesh = target_ax.pcolormesh(
            x2d, y2d, bg_frames[0], cmap=use_cmap, shading="gouraud", vmin=vmin, vmax=vmax, zorder=1
        )
        cbar = fig.colorbar(bg_mesh, ax=target_ax, fraction=0.046, pad=0.04)
        if bg_str in {"vorticity", "curl", "w"}:
            cbar.set_label(r"$\omega_z\ [\mathrm{s}^{-1}]$", fontsize=10)
        elif bg_str in {"mag", "speed"}:
            cbar.set_label(r"$\|\mathbf{u}\|\ [\mathrm{m/s}]$", fontsize=10)
    else:
        bg_mesh = None

    # Quiver Vector Layer
    ny, nx = y2d.shape
    if skip is None:
        step_y = step_x = max(1, int(round(max(nx, ny) / 16)))
    elif isinstance(skip, (tuple, list)):
        step_y, step_x = max(1, int(skip[0])), max(1, int(skip[1]))
    else:
        step_y = step_x = max(1, int(skip))
    dx_step = abs(float(x_arr[1] - x_arr[0])) if len(x_arr) > 1 else 1.0
    dy_step = abs(float(y_arr[1] - y_arr[0])) if len(y_arr) > 1 else 1.0

    def _apply_chc_mask(ds_t, *arrs):
        if "chc" not in ds_t:
            return arrs
        invalid = np.asarray(ds_t["chc"].values, dtype=float) == 0
        return tuple(np.where(invalid, np.nan, a) for a in arrs)

    ds0 = data.isel(t=0) if "t" in data.dims else data
    u0 = np.asarray(ds0["u"].values, dtype=float)
    v0 = np.asarray(ds0["v"].values, dtype=float)
    u0, v0 = _apply_chc_mask(ds0, u0, v0)
    med_speed = float(np.nanmedian(np.sqrt(u0**2 + v0**2)))
    if med_speed == 0.0:
        med_speed = 1.0

    if arrow_scale is None:
        target_len = 0.85 * min(step_x * dx_step, step_y * dy_step)
        auto_scale = (med_speed / target_len) if target_len > 0 else 1.0
    else:
        auto_scale = float(arrow_scale)

    def color_frame(ds_t, u_t, v_t):
        cb_str = str(color_by).lower()
        if cb_str in ("mag", "magnitude", "speed"):
            arr = np.sqrt(u_t**2 + v_t**2)
        elif str(color_by) in ds_t:
            arr = np.asarray(ds_t[str(color_by)].values, dtype=float)
            arr, = _apply_chc_mask(ds_t, arr)
        else:
            warnings.warn(f"color_by={color_by!r} not found in dataset; using arrow_color instead")
            return None
        return arr

    color0 = color_frame(ds0, u0, v0) if color_by is not None else None
    quiver_extra = {"color": arrow_color} if color0 is None else {
        "cmap": use_cmap if use_cmap is not None else "viridis",
    }

    if quiver:
        Q = target_ax.quiver(
            x2d[::step_y, ::step_x],
            y2d[::step_y, ::step_x],
            u0[::step_y, ::step_x],
            v0[::step_y, ::step_x],
            *(() if color0 is None else (color0[::step_y, ::step_x],)),
            angles="xy",
            scale_units="xy",
            scale=auto_scale,
            width=float(arrow_width),
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.5,
            minshaft=1.5,
            pivot="mid",
            alpha=float(arrow_alpha),
            zorder=2,
            **quiver_extra,
        )
        if color0 is not None:
            fig.colorbar(Q, ax=target_ax, fraction=0.046, pad=0.04).set_label(str(color_by), fontsize=10)
    else:
        Q = None

    target_ax.set_aspect("equal")
    x_units = str(data.attrs.get("spatial_units", data.coords.get("x", {}).attrs.get("units", "mm")))
    y_units = str(data.attrs.get("spatial_units", data.coords.get("y", {}).attrs.get("units", "mm")))
    target_ax.set_xlabel(f"x [{x_units}]", fontsize=10)
    target_ax.set_ylabel(f"y [{y_units}]", fontsize=10)

    t0_val = float(ds0["t"].values) if ("t" in ds0.coords and ds0["t"].size) else 0.0
    target_ax.set_title(title_fmt.format(t=t0_val, i=0), fontsize=11, fontweight="bold")
    fig.tight_layout()

    def update_frame(frame_i):
        artists = []
        if bg_mesh is not None and frame_i < len(bg_frames):
            bg_mesh.set_array(bg_frames[frame_i])
            artists.append(bg_mesh)

        ds_t = data.isel(t=frame_i) if "t" in data.dims else data
        if Q is not None:
            u_t = np.asarray(ds_t["u"].values, dtype=float)
            v_t = np.asarray(ds_t["v"].values, dtype=float)
            u_t, v_t = _apply_chc_mask(ds_t, u_t, v_t)
            if color0 is not None:
                c_t = color_frame(ds_t, u_t, v_t)
                Q.set_UVC(u_t[::step_y, ::step_x], v_t[::step_y, ::step_x], c_t[::step_y, ::step_x])
            else:
                Q.set_UVC(u_t[::step_y, ::step_x], v_t[::step_y, ::step_x])
            artists.append(Q)

        t_coord = float(ds_t["t"].values) if ("t" in ds_t.coords and ds_t["t"].size) else float(frame_i)
        target_ax.set_title(title_fmt.format(t=t_coord, i=frame_i), fontsize=11, fontweight="bold")
        return artists if blit else []

    anim = FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=interval,
        repeat=repeat,
        blit=blit,
    )
    return anim
