import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

DEFAULT_FIELD_INTERPOLATION_ORDER=3

default_rng = np.random.default_rng()


def _field_interpolator(field, x_old, y_old, interpolation_order=DEFAULT_FIELD_INTERPOLATION_ORDER):

    spline = RectBivariateSpline(y_old, x_old, field,
                                 kx=interpolation_order, ky=interpolation_order)

    return lambda x, y: spline(y, x)[0,0]


def seed_generator(n_seeds, ranges, rng=default_rng):
    range_x, range_y = ranges
    x0s = rng.random(size=n_seeds) * (range_x[1] - range_x[0]) + range_x[0]
    y0s = rng.random(size=n_seeds) * (range_y[1] - range_y[0]) + range_y[0]
    return np.array([x0s, y0s]).T


def integrate_trajectories(
    t: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    u: np.ndarray, v: np.ndarray,
    initial_coordinates: np.ndarray, n_steps: int,
    interpolation_order=DEFAULT_FIELD_INTERPOLATION_ORDER):

    u = np.where(np.isnan(u), 0.0, u)
    v = np.where(np.isnan(v), 0.0, v)

    dt = t[1] - t[0]
    n_lines = len(initial_coordinates)


    lines = np.zeros((n_lines, n_steps, 2))
    for j, (x0, y0) in enumerate(initial_coordinates):
        lines[j][0][0] = x0
        lines[j][0][1] = y0


    for i in range(n_steps-1):
        u_interpolator = _field_interpolator(u[i], x, y, interpolation_order)
        v_interpolator = _field_interpolator(v[i], x, y, interpolation_order)


        for j in range(n_lines):
            lines[j][i+1][0] = lines[j][i][0] + u_interpolator(*lines[j][i]) * dt
            lines[j][i+1][1] = lines[j][i][1] + v_interpolator(*lines[j][i]) * dt


    return lines


if __name__ == "__main__":
    Nt = 20
    shape = (50, 50)
    t = np.linspace(0, 1, Nt)

    x_coord = np.linspace(0, 1, shape[1])
    y_coord = np.linspace(0, 1, shape[0])

    x, y = np.meshgrid(x_coord, y_coord)
    u, v = np.array([-y]*Nt), np.array([x]*Nt)

    initial_coordinates = seed_generator(5, 1, shape)

    plt.plot(initial_coordinates[:,0], initial_coordinates[:,1])
    plt.quiver(x, y, u[0], v[0])
    plt.show()

    lines = integrate_trajectories(t, x_coord, y_coord, u, v, initial_coordinates, n_steps=Nt)

    for line in lines:
        plt.plot(line[:,0], line[:,1])

    plt.show()
