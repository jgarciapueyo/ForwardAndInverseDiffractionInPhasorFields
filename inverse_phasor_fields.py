from pathlib import Path
from typing import List

import matplotlib as mpl
import numpy as np
from PIL import Image
import tal
from tal.enums import CameraSystem

import utils_diffraction as udiff
import utils_diffraction_inverse as udiffinv
import utils_nlos as unlos


def fromarray(x, sampling_points_x, sampling_points_y):
    """Transforms a NLOS reconstruction into an image."""
    cmap = mpl.colormaps["hot"]
    abs_x = np.abs(x).reshape((sampling_points_x, sampling_points_y))
    max = np.max(abs_x)
    normalised = abs_x / max
    im = Image.fromarray(np.uint8(cmap(normalised) * 255))
    return im


def print_input_output_plane_params(
    aperture_size: float,
    sampling_points: int,
    distance: float,
    wavelength: float,
    distance_between_samples: float,
    min_wv: float,
    max_sp_freq_wv: float,
    max_sp_freq_shannon: float,
    max_sp_freq_fap: float,
):
    """Prints some information about the input and output planes including the maximum spatial
    frequencies that are conserved when propagating."""
    print(
        f"A ({aperture_size}) = #s ({sampling_points}) * Δs ({distance_between_samples})"
    )
    print(f"distance:   {distance}")
    print(f"wavelength: {wavelength}")
    print(20 * "=")
    print(f"Min wv: {min_wv}")
    print(f"Max sp freq wv: {max_sp_freq_wv}")
    print(f"Max sp freq Shannon: {max_sp_freq_shannon}")
    print(f"Max sp freq fap: {max_sp_freq_fap}")


def reconstruct_single_wavelength_forward(
    filename: str, depths: List[float], wl_mean: float
) -> np.ndarray:
    """
    Reconstructs the hidden object at different depths using the forward diffraction operator
    (equivalent to the reciprocity theorem as described by Eq. 22).
     The with virtual illumination pulse has a single wavelength defined by wl_mean.

    Parameters
    ----------
    filename : str
        Path to the data to load
    depths : List[float]
        List of depths at which to reconstruct the hidden object
    wl_mean : float
        Wavelength of the virtual illumination pulse (in time-frequency domain) as defined in
         Phasor Fields

    Returns
    -------
    The reconstruction at the specified depths.
    """
    data = tal.io.read_capture(filename)
    wavefactor, wave_cycles = unlos.wavelength_to_wavefactor(wl_mean, 0, data)
    volume = tal.reconstruct.get_volume_project_rw(data, depths=depths)

    with tal.resources("max"):
        reconstructions_pf = tal.reconstruct.pf.solve(
            data,
            wavefactor=wavefactor,
            wave_cycles=wave_cycles,
            volume=volume,
            verbose=0,
        )

    return reconstructions_pf


def save_single_wavelength_forward(
    filename: str, depths: List[float], wl_mean: float, reconstructions: np.ndarray
):
    """Saves a reconstruction in array format into an image."""
    name = Path(filename).stem
    path = f"results/{name}/single_wavelength_forward/"
    Path(path).mkdir(parents=True, exist_ok=True)

    for i_d, d in enumerate(depths):
        np.save(path + f"depth{d}_wlmean{wl_mean}.npy", reconstructions[:, :, i_d])
        im = fromarray(
            reconstructions[:, :, i_d],
            reconstructions.shape[0],
            reconstructions.shape[1],
        )
        im.save(path + f"depth{d}_wlmean{wl_mean}.png")


def reconstruct_multi_wavelength_forward(
    filename: str, depths: List[float], wl_mean: float, wl_sigma: float
) -> np.ndarray:
    """
    Reconstructs the hidden object at different depths using the forward diffraction operator
    (equivalent to the reciprocity theorem as described by Eq. 22).
     The with virtual illumination pulse has a gaussian distribution of time-frequencies with
     central wavelength defined by wl_mean and standard deviation defined by wl_sigma.

    Parameters
    ----------
    filename : str
        Path to the data to load
    depths : List[float]
        List of depths at which to reconstruct the hidden object
    wl_mean : float
        Wavelength of the virtual illumination pulse (in time-frequency domain) as defined in
         Phasor Fields
    wl_sigma : float
        Standard deviation of the gaussian pulse dfininin the virtual illumination pulse (in
        time-frequency domain)

    Returns
    -------
    The reconstruction at the specified depths.
    """
    data = tal.io.read_capture(filename)
    volume = tal.reconstruct.get_volume_project_rw(data, depths=depths)

    with tal.resources("max"):
        reconstructions_fbp = tal.reconstruct.fbp.solve(
            data,
            wl_mean=wl_mean,
            wl_sigma=wl_sigma,
            border="zero",
            volume_xyz=volume,
            volume_format=data.volume_format,
            camera_system=CameraSystem.DIRECT_LIGHT,
            progress=False,
        )

    return reconstructions_fbp


def save_multi_wavelength_forward(
    filename: str,
    depths: List[float],
    wl_mean: float,
    wl_sigma: float,
    reconstructions: np.ndarray,
):
    """Saves a reconstruction in array format into an image."""
    name = Path(filename).stem
    path = f"results/{name}/multi_wavelength_forward/wlsigma{wl_sigma}/"
    Path(path).mkdir(parents=True, exist_ok=True)

    for i_d, d in enumerate(depths):
        np.save(path + f"depth{d}_wlmean{wl_mean}.npy", reconstructions[:, :, i_d])
        im = fromarray(
            reconstructions[:, :, i_d],
            reconstructions.shape[0],
            reconstructions.shape[1],
        )
        im.save(path + f"depth{d}_wlmean{wl_mean}.png")


def reconstruct_single_wavelength_inverse(
    filename: str, depths: List[float], wl_mean: float, pinv_reg: float
):
    """
    Reconstructs the hidden object at different depths using the inverse diffraction operators
    described in Table 1: pseudoinverse/reciprocal in Eq.18/19, adjoint in Eq. 20/21,
    and reciprocity theorem in Eq. 22/23. The with virtual illumination pulse has a single
    wavelength defined by wl_mean.

    These reconstructions correspond to Fig. 4 and Fig. 5 (for different pinv_reg values).

    Parameters
    ----------
    filename : str
        Path to the data to load
    depths : List[float]
        List of depths at which to reconstruct the hidden object
    wl_mean : float
        Wavelength of the virtual illumination pulse (in time-frequency domain) as defined in
         Phasor Fields
    pinv_reg: float
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)

    Returns
    -------
    A dictionary with the reconstructions at different depths, the forward diffraction operator
    and the inverse diffraction operators.
    """
    reconstructions = {
        "spatial_pinv": [],
        "spatial_adj": [],
        "spatial_rec": [],
        "spatialfreq_reciprocal": [],
        "spatialfreq_adj": [],
        "spatialfreq_rec": [],
    }
    g_z_list = []
    g_z_pinv_list = []

    for depth in depths:
        # Read the data
        data = tal.io.read_capture(filename)
        data.downscale(2)

        # Compute some variables of the setup
        aperture_size_x, aperture_size_y = unlos.compute_aperture_size(data)
        sampling_points_x, sampling_points_y = unlos.compute_sampling_points(data)
        distance_between_samples = unlos.compute_distance_between_samples(data)
        wavefactor, wave_cycles = unlos.wavelength_to_wavefactor(wl_mean, 0, data)
        input_plane, output_plane = unlos.define_input_output_planes(data, depth)
        k = 2 * np.pi / wl_mean
        min_wv = udiff.min_wavelength(distance_between_samples)
        max_sp_freq_wv = udiff.max_spatial_frequency_wavelength(wl_mean)
        max_sp_freq_shannon = udiff.max_spatial_frequency_shannon(
            distance_between_samples
        )
        max_sp_freq_fap = udiff.max_spatial_frequency_finiteap(
            aperture_size_x, wl_mean, depth
        )

        if False:
            # Prints information about the setup
            print_input_output_plane_params(
                aperture_size_x,
                sampling_points_x,
                depth,
                wl_mean,
                distance_between_samples,
                min_wv,
                max_sp_freq_wv,
                max_sp_freq_shannon,
                max_sp_freq_fap,
            )

        # 1. Virtual illumination of the impulse response function (in data.H)
        # This is equivalent to Eq. 10 (in time-frequency domain).
        # After virtual illumination P[x_l, t], which is equivalent to Eq. 10 in time-frequency
        # domain, we get s_omega which represents the output wavefield in time-frequency spatial
        # domain.
        s_omega_img, wavelength, (illumination_pulse, _, _) = (
            tal.reconstruct.pf.to_Fourier(
                data, wavefactor=wavefactor, pulse_cycles=wave_cycles
            )
        )
        s_omega_img = s_omega_img[0]  # Needed because of dimensions returned
        # 1.1. Transform s_omega into single vector (equivalent to u_z)
        s_omega_vector = s_omega_img.reshape((sampling_points_x * sampling_points_y, 1))
        u_z_vector = np.conj(s_omega_vector)
        # 1.2. Compute S_omega which represents the output wavefield in time-frequency
        # spatial-frequency domain
        S_omega, freqx, freqy, spatial_frecuency_stepx, spatial_frecuency_stepy = (
            udiff.ft(s_omega_img, distance_between_samples)
        )
        # 1.3. Transform into equivalent U_z
        U_z = np.conj(S_omega)

        # 2. Compute forward diffraction operator in:
        # 2.1. Spatial domain (Eq. 2)
        g_z = udiff.forward_diffraction_operator_spatial(
            input_plane, output_plane, wl_mean
        )
        # 2.2. Spatial-frequency domain (Eq. 4)
        G_z, mask = udiff.forward_diffraction_operator_spatialfrequency(
            input_plane, output_plane, wl_mean, freqx, freqy
        )

        # 3. Reconstruct hidden object using inverse diffraction in spatial form
        #   (Eq. 18, 20, 22 applied to s_omega/u_z)
        u_0_estimated_pinv, u_0_estimated_adj, u_0_estimated_rec, g_z_pinv = (
            udiffinv.inverse_diffraction_spatial_form(u_z_vector, g_z, pinv_reg)
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

        # 4. Reconstruct hidden object using inverse diffraction in spatial-frequency form
        #   (Eq. 19, 21, 23 applied to s_omega/u_z)
        U_0_estimated_reciprocal, U_0_estimated_adj, U_0_estimated_rec = (
            udiffinv.inverse_diffraction_spatialfrequency_forms(U_z, G_z, mask)
        )
        u_0_estimated_reciprocal, _, _, _, _ = udiff.ift(
            U_0_estimated_reciprocal,
            freqx,
            freqy,
            spatial_frecuency_stepx,
            spatial_frecuency_stepy,
        )
        reconstructions["spatialfreq_reciprocal"].append(u_0_estimated_reciprocal)
        u_0_estimated_adj, _, _, _, _ = udiff.ift(
            U_0_estimated_adj,
            freqx,
            freqy,
            spatial_frecuency_stepx,
            spatial_frecuency_stepy,
        )
        reconstructions["spatialfreq_adj"].append(u_0_estimated_adj)
        u_0_estimated_rec, _, _, _, _ = udiff.ift(
            U_0_estimated_rec,
            freqx,
            freqy,
            spatial_frecuency_stepx,
            spatial_frecuency_stepy,
        )
        reconstructions["spatialfreq_rec"].append(u_0_estimated_rec)

        g_z_list.append(g_z)
        g_z_pinv_list.append(g_z_pinv)

    return reconstructions, g_z_list, g_z_pinv_list


def save_single_wavelength_inverse(
    filename: str,
    depths: List[float],
    wl_mean: float,
    pinv_reg: float,
    reconstructions,
    g_z_list: List[np.ndarray],
    g_z_pinv_list: List[np.ndarray],
):
    """Saves a reconstruction into an image."""
    name = Path(filename).stem
    path = f"results/{name}/single_wavelength_inverse/pinv_reg{pinv_reg}/"
    Path(path).mkdir(parents=True, exist_ok=True)

    for i_d, d in enumerate(depths):
        for type_inv in [
            "spatial_pinv",
            "spatial_adj",
            "spatial_rec",
            "spatialfreq_reciprocal",
            "spatialfreq_adj",
            "spatialfreq_rec",
        ]:
            np.save(
                path + f"{type_inv}_depth{d}_wlmean{wl_mean}.npy",
                reconstructions[type_inv][i_d],
            )
            im = fromarray(
                reconstructions[type_inv][i_d],
                reconstructions[type_inv][i_d].shape[0],
                reconstructions[type_inv][i_d].shape[1],
            )
            im.save(path + f"{type_inv}_depth{d}_wlmean{wl_mean}.png")
        np.save(path + f"gz_depth{d}_wlmean{wl_mean}.npy", g_z_list[i_d])
        np.save(path + f"gzpinv_depth{d}_wlmean{wl_mean}.npy", g_z_pinv_list[i_d])


def reconstruct_multi_wavelength_inverse(
    filename: str, depths: List[float], wl_mean: float, pinv_reg: float
):
    """
    Reconstructs the hidden object at different depths using the inverse diffraction operators
    described in Table 1: pseudoinverse/reciprocal in Eq.18/19, adjoint in Eq. 20/21,
    and reciprocity theorem in Eq. 22/23. The with virtual illumination pulse has a 5 wavelengths
    defined as the central wavelength (defined by wl_mean) and 4 other wavelengths
    delta_wl = (-0.02, -0.01, +0.01, +0.02).

    These reconstructions correspond to Fig. 6.

    Parameters
    ----------
    filename : str
        Path to the data to load
    depths : List[float]
        List of depths at which to reconstruct the hidden object
    wl_mean : float
        Wavelength of the virtual illumination pulse (in time-frequency domain) as defined in
         Phasor Fields
    pinv_reg: float
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)

    Returns
    -------
    A dictionary with the reconstructions at different depths, the forward diffraction operator
    and the inverse diffraction operators."
    """
    wl_mean_multi_list = [
        wl_mean - 0.02,
        wl_mean - 0.01,
        wl_mean + 0.01,
        wl_mean + 0.02,
    ]
    reconstructions_list = []
    g_z_multi_list = []
    g_z_pinv_multi_list = []

    for wl_mean_multi in wl_mean_multi_list:
        (
            reconstructions,
            g_z_list,
            g_z_pinv_list,
        ) = reconstruct_single_wavelength_inverse(
            filename, depths, wl_mean_multi, pinv_reg
        )
        reconstructions_list.append(reconstructions)
        g_z_multi_list.append(g_z_list)
        g_z_pinv_multi_list.append(g_z_pinv_list)

    return reconstructions_list, g_z_multi_list, g_z_pinv_multi_list


def save_multi_wavelength_inverse(
    filename: str,
    depths: List[float],
    wl_mean: float,
    pinv_reg: float,
    reconstructions_multi_wl_inverse,
    g_z_multi_list,
    g_z_pinv_multi_list,
):
    """Saves a reconstruction into an image."""
    name = Path(filename).stem

    wl_mean_multi_list = [
        wl_mean - 0.02,
        wl_mean - 0.01,
        wl_mean + 0.01,
        wl_mean + 0.02,
    ]

    for i_wl, wl_mean_multi in enumerate(wl_mean_multi_list):
        path = f"results/{name}/multi_wavelength_inverse/pinv_reg{pinv_reg}/wlmean{wl_mean_multi}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        for i_d, d in enumerate(depths):
            for type_inv in [
                "spatial_pinv",
                "spatial_adj",
                "spatial_rec",
                "spatialfreq_reciprocal",
                "spatialfreq_adj",
                "spatialfreq_rec",
            ]:
                np.save(
                    path + f"{type_inv}_depth{d}_wlmean{wl_mean}.npy",
                    reconstructions_multi_wl_inverse[i_wl][type_inv][i_d],
                )
                im = fromarray(
                    reconstructions_multi_wl_inverse[i_wl][type_inv][i_d],
                    reconstructions_multi_wl_inverse[i_wl][type_inv][i_d].shape[0],
                    reconstructions_multi_wl_inverse[i_wl][type_inv][i_d].shape[1],
                )
                im.save(path + f"{type_inv}_depth{d}_wlmean{wl_mean}.png")
            np.save(
                path + f"gz_depth{d}_wlmean{wl_mean}.npy", g_z_multi_list[i_wl][i_d]
            )
            np.save(
                path + f"gzpinv_depth{d}_wlmean{wl_mean}.npy", g_z_pinv_multi_list[i_d]
            )


def reconstruct(
    filename: str,
    depths: List[float],
    wl_mean_list: List[float],
    wl_sigma_list: List[float],
    pinv_reg_list: List[float],
):
    """
    Reconstructs the hidden scene using forward and inverse Phasor Fields methods using
    single and multiple virtual wavelengths at different depths. It saves the reconstructions
    in the `results/` folder.

    Parameters
    ----------
    filename : str
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
    pinv_reg_list : List[float]
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    """
    for wl_mean in wl_mean_list:
        try:
            reconstructions_single_wl_forward = reconstruct_single_wavelength_forward(
                filename, depths, wl_mean
            )
            save_single_wavelength_forward(
                filename, depths, wl_mean, reconstructions_single_wl_forward
            )
        except Exception as e:
            print(f"Error: single_wavelength_forward - wl_mean: {wl_mean}")
            print(e)

        for wl_sigma in wl_sigma_list:
            try:
                reconstructions_multi_wl_forward = reconstruct_multi_wavelength_forward(
                    filename, depths, wl_mean, wl_sigma
                )
                save_multi_wavelength_forward(
                    filename,
                    depths,
                    wl_mean,
                    wl_sigma,
                    reconstructions_multi_wl_forward,
                )
            except Exception as e:
                print(
                    f"Error: single_wavelength_forward - wl_mean: {wl_mean}, wl_sigma: {wl_sigma}"
                )
                print(e)

        for pinv_reg in pinv_reg_list:
            try:
                reconstructions_single_wl_inverse, g_z_list, g_z_pinv_list = (
                    reconstruct_single_wavelength_inverse(
                        filename, depths, wl_mean, pinv_reg
                    )
                )
                save_single_wavelength_inverse(
                    filename,
                    depths,
                    wl_mean,
                    pinv_reg,
                    reconstructions_single_wl_inverse,
                    g_z_list,
                    g_z_pinv_list,
                )
            except Exception as e:
                print(
                    f"Error: single_wavelength_inverse - wl_mean: {wl_mean}, pinv_reg: {pinv_reg}"
                )
                print(e)
                pass

        for pinv_reg in pinv_reg_list:
            try:
                (
                    reconstructions_multi_wl_inverse,
                    g_z_multi_list,
                    g_z_pinv_multi_list,
                ) = reconstruct_multi_wavelength_inverse(
                    filename, depths, wl_mean, pinv_reg
                )
                save_multi_wavelength_inverse(
                    filename,
                    depths,
                    wl_mean,
                    pinv_reg,
                    reconstructions_multi_wl_inverse,
                    g_z_multi_list,
                    g_z_pinv_multi_list,
                )
            except Exception as e:
                print(
                    f"Error: multi_wavelength_inverse - wl_mean: {wl_mean}, pinv_reg: {pinv_reg}"
                )
                print(e)


if __name__ == "__main__":
    #===========================================================
    # 1. Reconstruct hidden scenes from real data (Fig. 4, Fig. 5, and Fig. 6)
    #    Attention: for loading real data (performat_letter4.mat), you must change file
    #                   venv/lib/python3.10/site-packages/tal/io/capture_data.py, line 246:
    #               from:
    #                   value = yaml.load(value, Loader=yaml.CLoader)
    #                to:
    #                   value = yaml.load(str(value), Loader=yaml.CLoader)
    #===========================================================
    folder = "data/pfdiffraction_fastnlos/tdata/"
    name = "performat_letter4.mat"
    # Distances from relay wall at which reconstruct the hidden scene
    depths = [1]
    # Wavelength mean for the virtual illumination pulse in time-frequency domain as described in
    # Phasor Fields
    wl_mean = [0.05]
    # Wavelength standard deviation for the virtual illumination pulse in time-frequency domain
    # as described in Phasor Fields
    wl_sigma = [0.2]
    # Parameter to regularize the pseudoinverse diffraction operator
    pinv_reg = [0.03, 0.1]
    reconstruct(folder + name, depths, wl_mean, wl_sigma, pinv_reg)

    #===========================================================
    # 2. Reconstruct hidden scenes from simulated data (Fig. 8a and 9a)
    #    The data for scenes from Fig. 8a and 9a are in data/simulated with name R_*m
    #       where * indicates a number between 1 and 10.
    #    You can also run `compuational_metric.py` to replicate those figures.
    #===========================================================
    folder = "data/simulated/R_1m/R_1m_s64_a2m/"
    name = "R_1m_s64_a2m.hdf5"
    # Distances from relay wall at which reconstruct the hidden scene
    depths = [1]
    # Wavelength mean for the virtual illumination pulse in time-frequency domain as described in
    # Phasor Fields
    wl_mean = [0.05]
    # Wavelength standard deviation for the virtual illumination pulse in time-frequency domain
    # as described in Phasor Fields
    wl_sigma = [0.2]
    # Parameter to regularize the pseudoinverse diffraction operator
    pinv_reg = [0.03, 0.1]
    reconstruct(folder + name, depths, wl_mean, wl_sigma, pinv_reg)

    #===========================================================
    # 3. Reconstruct hidden scenes from simulated data (Fig. 10)
    #    The data for scenes from Fig. 10 are in data/simulated/occlussions with name
    #        letters_S, letters_M, and letters_S_M
    #===========================================================
    folder = "data/simulated/occlusions/letters_S/"
    name = "letters_S.hdf5"
    # Distances from relay wall at which reconstruct the hidden scene
    depths = [1, 1.5]
    # Wavelength mean for the virtual illumination pulse in time-frequency domain as described in
    # Phasor Fields
    wl_mean = [0.05, 0.1]
    # Wavelength standard deviation for the virtual illumination pulse in time-frequency domain
    # as described in Phasor Fields
    wl_sigma = [0.2]
    # Parameter to regularize the pseudoinverse diffraction operator
    pinv_reg = [0.03, 0.1]
    reconstruct(folder + name, depths, wl_mean, wl_sigma, pinv_reg)
