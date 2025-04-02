import matplotlib.pyplot as plt
import numpy as np
from tal.io.capture_data import NLOSCaptureData


def define_input_output_planes(data: NLOSCaptureData, z: float, plot: bool = False):
    """
    Return the coordinates of the input and output planes.

    Parameters:
    -----------
    data : NLOSCaptureData
        Data representing the NLOS setup and the time-resolved data captured.
    z : float
        Distance in meters between input plane at z=0 and output plane at z.
        To correctly reconstruct the NLOS hidden scene, z must be equal to the
        distance where the hidden object is from the relay wall.
    plot : bool
        If `True`, it plots the input and output planes.

    Returns
    -------
    A tuple with the coordinates of the input and output planes.
    Each coordinate array has shape [sampling_points, sampling_points, 3].
    """
    input_plane_coords = data.sensor_grid_xyz
    output_plane_coords = input_plane_coords + np.array((0, 0, z))

    if plot:
        input_plane_coords_xs = input_plane_coords[:, :, 0]
        input_plane_coords_ys = input_plane_coords[:, :, 1]
        input_plane_coords_zs = input_plane_coords[:, :, 2]

        output_plane_coords_xs = output_plane_coords[:, :, 0]
        output_plane_coords_ys = output_plane_coords[:, :, 1]
        output_plane_coords_zs = output_plane_coords[:, :, 2]

        ax = plt.axes(projection="3d")
        ax.scatter(
            input_plane_coords_xs,
            input_plane_coords_ys,
            input_plane_coords_zs,
            label="$u_0(x,y)$",
        )
        ax.scatter(
            output_plane_coords_xs,
            output_plane_coords_ys,
            output_plane_coords_zs,
            label="$u_z(x,y)$",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    return input_plane_coords, output_plane_coords


def compute_distance_between_samples(data: NLOSCaptureData):
    """Computes the distance between scanned points of the relay wall,
    which is equivalent to distance between samples of a wavefield."""
    return np.linalg.norm(data.sensor_grid_xyz[0, 0, :] - data.sensor_grid_xyz[0, 1, :])


def compute_sampling_points(data: NLOSCaptureData):
    """Computes the number of scanned points of the relay wall at x and y axis."""
    return data.sensor_grid_xyz.shape[0], data.sensor_grid_xyz.shape[1]


def compute_aperture_size(data: NLOSCaptureData):
    """Computes the size of the aperture in x and y axis."""
    top_left_corner = data.sensor_grid_xyz[0, 0]
    top_right_corner = data.sensor_grid_xyz[-1, 0]
    bottom_left_corner = data.sensor_grid_xyz[0, -1]

    return np.linalg.norm(top_right_corner - top_left_corner), np.linalg.norm(
        bottom_left_corner - top_left_corner
    )


def wavefactor_to_wavelength(
    wavefactor: float, wave_cycles: float, data: NLOSCaptureData
):
    """Computes the equivalent virtual illumination pulse defined as wavelength mean and
    standard deviation from its definition as wavefactor and wave cycles."""
    wl_mean = wavefactor * np.linalg.norm(
        data.sensor_grid_xyz[0, 0, :] - data.sensor_grid_xyz[0, 1, :]
    )
    wl_sigma = wave_cycles * wl_mean / 6
    return wl_mean, wl_sigma


def wavelength_to_wavefactor(wl_mean: float, wl_sigma: float, data: NLOSCaptureData):
    """Computes the equivalent virtual illumination pulse defined as wavefactor and wave cycles
    from its definition as wavelength mean and standard deviation."""
    wavefactor = wl_mean / compute_distance_between_samples(data)
    wavecycles = 6 * wl_sigma / wl_mean
    return wavefactor, wavecycles
