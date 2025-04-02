from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy


def define_input_output_planes(
    z: float,
    aperture_size: float,
    sampling_points: int,
    plot: bool = False,
    rotation_input: Tuple[str, float] = None,
    rotation_output: Tuple[str, float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the coordinates of the input and output planes.

    Parameters:
    -----------
    z : float
        Distance in meters between input plane at z=0 and output plane at z.
    aperture_size : float
        Length in meters of the input and output planes.
    sampling_points : int
        Number of sampling points of the input and output planes in each direction.
        Total number of sampling points is sampling_points^2.
    plot : bool
        If `True`, it plots the input and output planes.
    rotation_input, rotation_output : Tuple[float, float]
        Axis and angle of rotation for input and output planes.

    Returns
    -------
    A tuple with the coordinates of the input and output planes.
    Each coordinate array has shape [sampling_points, sampling_points, 3].
    """
    x = np.linspace(-aperture_size / 2, aperture_size / 2, sampling_points)
    y = np.linspace(-aperture_size / 2, aperture_size / 2, sampling_points)
    xv, yv = np.meshgrid(x, y)

    input_plane_coords = np.stack(
        [xv, yv, np.zeros([sampling_points, sampling_points])]
    ).T
    output_plane_coords = input_plane_coords + np.array((0, 0, z))

    if rotation_input is not None:
        R = scipy.spatial.transform.Rotation.from_euler(
            rotation_input[0], rotation_input[1], degrees=True
        ).as_matrix()
        input_plane_coords = input_plane_coords @ R

    if rotation_output is not None:
        R = scipy.spatial.transform.Rotation.from_euler(
            rotation_output[0], rotation_output[1], degrees=True
        ).as_matrix()
        output_plane_coords = output_plane_coords @ R

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


def _input_output_plane_distance(
    input_plane: np.ndarray, output_plane: np.ndarray, for_loop: bool = False
) -> np.ndarray:
    """
    Computes the distance between all pair of sampling points in the input and output planes.

    Parameters
    ----------
    input_plane, output_plane : np.ndarray
        Coordinates of the input and output planes
    for_loop : bool
        If `True`, it uses a slower version using a for loop.

    Returns
    -------
    Distances between all pair of sampling points in the input and output planes.
    """
    input_plane_flatten = input_plane.reshape(-1, 3)
    output_plane_flatten = output_plane.reshape(-1, 3)

    if for_loop:
        x, y = input_plane.shape[0], input_plane.shape[1]
        distance = np.zeros((x * y, x * y))
        for i in range(input_plane.shape[0]):
            for j in range(input_plane.shape[1]):
                distance[i][j] = np.linalg.norm(
                    output_plane_flatten[j] - input_plane_flatten[i], ord=None, axis=-1
                )
    else:
        dd = np.expand_dims(output_plane_flatten, axis=1) - np.expand_dims(
            input_plane_flatten, axis=0
        )
        distance = np.linalg.norm(dd, ord=None, axis=-1)
    return distance


def forward_diffraction_operator_spatial(
    input_plane: np.ndarray, output_plane: np.ndarray, wavelength: float
):
    """
    Computes the forward diffraction operator g_z (Eq. 2) in spatial domain.

    Parameters
    ----------
    input_plane, output_plane : np.ndarray
        Coordinates of the input and output planes
    wavelength : float
        Wavelength (λ) of the input wavefield being propagated by the operator.

    Returns
    -------
    The forward diffraction operator g_z (Eq. 2) in spatial domain for each point (x,y) in the input plane.
    It has shape [sampling_points*sampling_points, sampling_points*sampling_points].
    """
    # Computes distance r from point i at the input plane to the point j at the output plane
    r = _input_output_plane_distance(input_plane, output_plane)
    # Wavenumber of the wavefield
    k = 2 * np.pi / wavelength
    # Distance in z direction between input plane and output plane
    z = output_plane[0, 0, 2]
    # Compute forward diffraction operator in spatial domain (Eq. 2)
    forward_diff_op_spatial = (
        (z / r) * (np.exp(1j * k * r) / r) * (1 / (1j * wavelength))
    )
    return forward_diff_op_spatial


def forward_diffraction_operator_spatialfrequency(
    input_plane: np.ndarray,
    output_plane: np.ndarray,
    wavelength: float,
    freqx: np.ndarray,
    freqy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward diffraction operator G_z (Eq. 4) in spatial-frequency domain.

    Parameters
    ----------
    input_plane, output_plane : np.ndarray
        Coordinates of the input and output planes.
    wavelength : float
        Wavelength (λ) of the input wavefield being propagated by the operator.
    freqx, freqy : np.ndarray
        Spatial frequencies of the wavefield in x and y dimension

    Returns
    -------
    The forward diffraction operator G_z (Eq. 4) in spatial-frequency domain for each spatial-frequency contained in the input wavefield.
    It has shape [freqx.len, freqy.len]. It also returns a mask indicating the homogeneous waves as `True`.
    """
    # Meshgrid of spatial frequencies
    fx, fy = np.meshgrid(freqy, freqx)
    # Wavenumber of the wavefield
    k = 2 * np.pi / wavelength
    # Distance in z direction between input plane and output plane
    z = output_plane[0, 0, 2]
    # Compute forward diffraction operator in spatial-frequency domain (Eq. 4)
    m_partial = (wavelength * fx) ** 2 + (wavelength * fy) ** 2
    mask = m_partial <= 1
    m = np.empty_like(m_partial, dtype=np.cdouble)
    m[mask] = np.sqrt(1 - m_partial[mask])
    m[~mask] = 1j * np.sqrt(m_partial[~mask] - 1)
    forward_diff_op_spatialfreq = np.exp(1j * k * z * m)
    return forward_diff_op_spatialfreq, mask


def ft(
    image: np.ndarray, distance_between_samples: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast Fourier Transform (FFT) of a 2D array (image).

    Parameters
    ----------
    image : np.ndarray
        2D array defining the image.
    distance_between_samples : float
        Distance between sampling points of the 2D image.

    Returns
    -------
    A tuple with the following elements:
    - the fast fourier transform of the image
    - the spatial-frequencies along x axis
    - the spatial-frequencies along y axis
    - the step between spatial-frequencies along x axis
    - the step between spatial-frequencies along y axis
    """
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    freqx = np.fft.fftshift(np.fft.fftfreq(image.shape[0], d=distance_between_samples))
    freqy = np.fft.fftshift(np.fft.fftfreq(image.shape[1], d=distance_between_samples))
    spatial_frecuency_stepx = freqx[1] - freqx[0]
    spatial_frecuency_stepy = freqy[1] - freqy[0]
    return image_fft, freqx, freqy, spatial_frecuency_stepx, spatial_frecuency_stepy


def ift(
    image_fft: np.ndarray,
    freqx: np.ndarray,
    freqy: np.ndarray,
    spatial_frecuency_stepx: np.ndarray,
    spatial_frecuency_stepy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse Fast Fourier Transform (IFFT) of a 2D array of spatial-frequencies.

    Parameters
    ----------
    image_fft : np.ndarray
        2D array defining the spatial-frequencies of an image.
    freqx, freqy : np.ndarray
        Spatial-frequencies along x and y axis.
    spatial_frecuency_stepx, spatial_frecuency_stepy : np.ndarray
        step between spatial-frequencies along x and y axis.

    Returns
    -------
    A tuple with the following elements:
    - the original image after applying the IFFT
    - the samples along x axis
    - the samples along y axis
    - the distance between samples along x axis
    - the distance between samples along y axis
    """
    image = np.fft.ifft2(np.fft.ifftshift(image_fft))
    samplesx = np.fft.ifftshift(np.fft.fftfreq(freqx.shape[0], spatial_frecuency_stepx))
    samplesy = np.fft.ifftshift(np.fft.fftfreq(freqy.shape[0], spatial_frecuency_stepy))
    distance_between_samplesx = samplesx[1] - samplesx[0]
    distance_between_samplesy = samplesy[1] - samplesy[0]
    return (
        image,
        samplesx,
        samplesy,
        distance_between_samplesx,
        distance_between_samplesy,
    )


def min_wavelength(distance_between_samples: float):
    """
    Minimum possible wavelength according to the Nyquist-Shannon sampling theorem
     applied to wavefields, λ > max(Δp, c * Δt) * 2, where λ is the wavelength of the
     wavefield, Δp is the distance between samples and Δt is the temporal resolution.
     This is applicable to NLOS wavefields produced by applying the FFT along the temporal
     domain to get "virtual" wavefields (also known as Phasor Fields).
    """
    return distance_between_samples * 2


def max_spatial_frequency_shannon(distance_between_samples: float):
    """
    Maximum spatial frequency that can be propagated without loss according
     to the distance between samples: fx < 1/(2*Δp)
    """
    return 1 / (distance_between_samples * 2)


def max_spatial_frequency_wavelength(wavelength: float):
    """
    Maximum spatial frequency that can be propagated without loss according to
     homogeneous waves: f_x**2 + f_y**2 < 1 / λ^2"""
    return 1 / (2 * (wavelength**2))


def max_spatial_frequency_finiteap(
    aperture_size: float, wavelength: float, distance: float
):
    """
    Maximum spatial frequency that can be propagated assuming finite appertures:
    fx = L / 2M, where L is the length of the aperture an M =~ λz.
    """
    M = wavelength * distance
    return aperture_size / (2 * M)  # might feet better if without 2
