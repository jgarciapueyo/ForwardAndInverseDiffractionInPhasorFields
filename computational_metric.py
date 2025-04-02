from pathlib import Path
import time
from typing import List

import numpy as np
import tal

import utils_diffraction as udiff
import utils_diffraction_inverse as udiffinv
import utils_nlos as unlos


def reconstruct(name: str, d: float, wl_mean: float, pr: float, asize: int, sp: int):
    """
    Reconstructs the hidden scene using inverse Phasor Fields with single virtual wavelengths at different depths.
    It saves the reconstructions in the `results/` folder.

    Parameters
    ----------
    name : str
        Path to the data to load. This should contain the time-resolved impulse response
        function of the hidden scene and other calibration parameters (3d position of the
        sensed points in the relay wall, etc.).
    d : float
        Depth at which to reconstruct the hidden object
    wl_mean : float
        Wavelength of the virtual illumination pulse (in time-frequency domain)
        as defined in Phasor Fields
    pr : float
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    asize: int
        Size (in m) of the relay wall. Must be the same of the simulated data in `name`.
    sp : int
        Number of sampling points in the relay wall.
    """
    sp_original = 128  # originally, data is simulated with a grid of 128x128 points in the relay wall
    filename = f"data/simulated/{name}_{d}m/{name}_{d}m_s{sp_original}_a{asize}m/{name}_{d}m_s{sp_original}_a{asize}m.hdf5"
    data = tal.io.read_capture(filename)

    # Downscale data from 128 sampling points (used to capture the data) to the desired points.
    original_sampling_points_x, original_sampling_points_y = (
        unlos.compute_sampling_points(data)
    )
    downscale = original_sampling_points_x // sp
    if downscale > 1:
        data.downscale(downscale)

    # Compute some variables of the setup
    aperture_size_x, aperture_size_y = unlos.compute_aperture_size(data)
    sampling_points_x, sampling_points_y = unlos.compute_sampling_points(data)
    distance_between_samples = unlos.compute_distance_between_samples(data)
    wavefactor, wave_cycles = unlos.wavelength_to_wavefactor(wl_mean, 0, data)
    input_plane, output_plane = unlos.define_input_output_planes(data, d)

    # 1. Virtual illumination of the impulse response function (in data.H)
    # This is equivalent to Eq. 10 (in time-frequency domain).
    # After virtual illumination P[x_l, t], which is equivalent to Eq. 10 in time-frequency
    # domain, we get s_omega which represents the output wavefield in time-frequency spatial
    # domain.
    s_omega_img, wavelength, (illumination_pulse, _, _) = tal.reconstruct.pf.to_Fourier(
        data, wavefactor=wavefactor, pulse_cycles=wave_cycles
    )
    s_omega_img = s_omega_img[0]  # Needed because of dimensions returned
    # 1.1. Transform s_omega into single vector (equivalent to u_z)
    s_omega_vector = s_omega_img.reshape((sampling_points_x * sampling_points_y, 1))
    u_z_vector = np.conj(s_omega_vector)
    # 1.2. Compute S_omega which represents the output wavefield in time-frequency
    # spatial-frequency domain
    S_omega, freqx, freqy, spatial_frecuency_stepx, spatial_frecuency_stepy = udiff.ft(
        s_omega_img, distance_between_samples
    )
    # 1.3. Transform into equivalent U_z
    U_z = np.conj(S_omega)

    # 2. Compute forward diffraction operator in:
    # 2.1. Spatial domain (Eq. 2)
    g_z = udiff.forward_diffraction_operator_spatial(input_plane, output_plane, wl_mean)
    # 2.2. Spatial-frequency domain (Eq. 4)
    G_z, mask = udiff.forward_diffraction_operator_spatialfrequency(
        input_plane, output_plane, wl_mean, freqx, freqy
    )
    # print_operators_info(input_plane, output_plane, g_z, G_z)

    reconstructions = {
        "spatial_pinv": [],
        "spatial_adj": [],
        "spatial_rec": [],
        "spatialfreq_reciprocal": [],
        "spatialfreq_adj": [],
        "spatialfreq_rec": [],
    }

    # 3. Reconstruct hidden object using inverse diffraction in spatial form
    #   (Eq. 18, 20, 22 applied to s_omega/u_z)
    u_0_estimated_pinv, u_0_estimated_adj, u_0_estimated_rec, g_z_pinv = (
        udiffinv.inverse_diffraction_spatial_form(u_z_vector, g_z, pr)
    )
    reconstructions["spatial_pinv"].append(
        u_0_estimated_pinv.reshape((sampling_points_x, sampling_points_y))
    )
    reconstructions["spatial_adj"].append(
        u_0_estimated_adj.reshape((sampling_points_x, sampling_points_y))
    )
    reconstructions["spatial_rec"].append(
        u_0_estimated_rec.reshape((sampling_points_x, sampling_points_y))
    )

    return reconstructions, g_z, g_z_pinv


def computational_metric(g_z: np.ndarray, g_z_pinv: np.ndarray, pr: float):
    """
    Computes the rank and condition number of the forward diffraction operator g_z
    in matrix form. For the rank, the tolerance is given by the pr (t_th in the paper),
    as explained in paragraph containing Eq. 27 and Eq. 28 for the computation of the
    pseudoinverse of matrix.
    """
    rank = np.linalg.matrix_rank(g_z, tol=pr)
    condition_number = np.linalg.cond(g_z)
    return rank, condition_number


def estimate_computational_metric_reconstruction_limit(
    name: str,
    depths: List[float],
    wl_mean_list: List[float],
    pinv_reg: List[float],
    aperture_size: List[int],
    sampling_points: List[int],
):
    """
    Estimates the computational metric for NLOS reconstruction limits (Eq. 29), based on the relative
    rank of the forward diffraction matrix g_z. The computational metric is estimated for:
     different depths z, wavelengths λ, aperture sizes of the
    relay wall A and number of sampling points N_s of the relay wall.
    It also reconstructs the letter R at different depths D, where always the reconstructed plane z is at
    that depth D.

    Parameters
    ----------
    name : str
        Path to the data to load. This should contain the time-resolved impulse response
        function of the hidden scene and other calibration parameters (3d position of the
        sensed points in the relay wall, etc.).
    depths : List[float]
        List of depths at which to reconstruct the hidden object
    wl_mean_list : List[float]
        List of wavelengths of the virtual illumination pulse (in time-frequency domain)
        as defined in Phasor Fields
    wl_sigma_list : List[float]
        List of standard deviation of the gaussian pulse dfinining the virtual illumination
        pulse (in time-frequency domain) as defined in Phasor Fields
    pinv_reg : List[float]
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    aperture_size : List[float]
        List of the aperture sizes of the relay wall. It indicates the length of a square relay wall.
    sampling_points : List[float]
        List of the different number of sampling points of the relay wall.
    """
    for sp in sampling_points:
        for asize in aperture_size:
            for wl_mean in wl_mean_list:
                for d in depths:
                    for pr in pinv_reg:
                        start = time.time()
                        reconstructions, g_z, g_z_pinv = reconstruct(
                            name, d, wl_mean, pr, asize, sp
                        )
                        rank, condition_number = computational_metric(g_z, g_z_pinv, pr)
                        to_save = {
                            "reconstructions": reconstructions,
                            # "g_z": g_z,
                            # "g_z_pinv": g_z_pinv,
                            "rank": rank,
                            "condition_number": condition_number,
                        }
                        filepath = f"results/R/R_{d}m_{sp}sp_a{asize}m/"
                        Path(filepath).mkdir(parents=True, exist_ok=True)
                        np.save(
                            filepath
                            + f"R_{d}m_{sp}sp_a{asize}m_{wl_mean}wl_{pr}pinvreg.npy",
                            to_save,
                        )
                        end = time.time()
                        print(
                            f"Processed: {filepath}R_{d}m_{sp}sp_a{asize}m_{wl_mean}wl_{pr}pinvreg - time:{end - start}"
                        )


if __name__ == "__main__":
    name = "R"

    # Distances z of the hidden object letter R from the relay wall
    # These are also the distances D at which reconstruct the hidden object
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Size of the relay wall A (similar to size of the aperture)
    aperture_size = [2, 3, 4]
    # Number of scanned points in the relay wall N_s
    sampling_points = [32]
    # Wavelength mean for the virtual illumination pulse in time-frequency domain as described in
    # Phasor Fields
    wl_mean = [0.01, 0.03, 0.06, 0.10, 0.13]
    # Parameter to regularize the pseudoinverse diffraction operator
    pinv_reg = [0.1]

    estimate_computational_metric_reconstruction_limit(
        name, depths, wl_mean, pinv_reg, aperture_size, sampling_points
    )
