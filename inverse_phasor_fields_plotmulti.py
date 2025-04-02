from pathlib import Path
from typing import List

import matplotlib as mpl
import numpy as np
from PIL import Image


def fromarray(x, sampling_points_x, sampling_points_y, f=np.abs, max_value=None):
    """Transforms a NLOS reconstruction into an image."""
    cmap = mpl.colormaps["hot"]
    x_reshape = x.reshape((sampling_points_x, sampling_points_y))
    processed_x = f(x_reshape)
    if max_value is None:
        max_value = np.max(processed_x)
    normalised = processed_x / max_value
    im = Image.fromarray(np.uint8(cmap(normalised)*255))
    return im


def plot_multi_wavelength(
    name: str,
    depths: List[float],
    wl_mean_list: List[float],
    pinv_reg_list: List[float],
):
    """
    Assembles a multi wavelength reconstruction of the hidden scene using previous
      single wavelength reconstructions (made by the `inverse_phasor_fields.py` file).

    Parameters
    ----------
    name : str
        Name of the scene. Used to load the single wavelength reconstructions.
    depths : List[float]
        List of depths at which to reconstruct the hidden object
    wl_mean_list : List[float]
        List of wavelengths of the virtual illumination pulse (in time-frequency domain)
        as defined in Phasor Fields
    pinv_reg_list : List[float]
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    """
    name = Path(name).stem
    f = lambda x: np.abs(x)

    for pinv_reg in pinv_reg_list:
        path_save_reconstruction = f'results/{name}/multi_wavelength_inverse/pinv_reg{pinv_reg}/'
        Path(path_save_reconstruction).mkdir(exist_ok=True, parents=True)

        for d in depths:
            for i_wl, wl_mean in enumerate(wl_mean_list):
                spatial_pinv = np.load(f'results/{name}/single_wavelength_inverse/pinv_reg{pinv_reg}/spatial_pinv_depth{d}_wlmean{wl_mean}.npy')
                spatial_adj = np.load(f'results/{name}/single_wavelength_inverse/pinv_reg{pinv_reg}/spatial_adj_depth{d}_wlmean{wl_mean}.npy')
                spatial_rec = np.load(f'results/{name}/single_wavelength_inverse/pinv_reg{pinv_reg}/spatial_rec_depth{d}_wlmean{wl_mean}.npy')

                wl_mean_multi_list = [
                    wl_mean - 0.02,
                    wl_mean - 0.01,
                    wl_mean + 0.01,
                    wl_mean + 0.02,
                ]
                for i_wl_mean_plus_sigma, wl_mean_plus_sigma in enumerate(wl_mean_multi_list):
                    wl_mean_plus_sigma = round(wl_mean_plus_sigma, 2)
                    spatial_pinv += np.load(f'results/{name}/multi_wavelength_inverse/pinv_reg{pinv_reg}/wlmean{wl_mean_plus_sigma}/spatial_pinv_depth{d}_wlmean{wl_mean}.npy')
                    spatial_adj += np.load(f'results/{name}/multi_wavelength_inverse/pinv_reg{pinv_reg}/wlmean{wl_mean_plus_sigma}/spatial_adj_depth{d}_wlmean{wl_mean}.npy')
                    spatial_rec += np.load(f'results/{name}/multi_wavelength_inverse/pinv_reg{pinv_reg}/wlmean{wl_mean_plus_sigma}/spatial_rec_depth{d}_wlmean{wl_mean}.npy')

                im_spatial_pinv = fromarray(spatial_pinv, spatial_pinv.shape[0], spatial_pinv.shape[1], f)
                im_spatial_adj = fromarray(spatial_adj, spatial_adj.shape[0], spatial_adj.shape[1], f)
                im_spatial_rec = fromarray(spatial_rec, spatial_rec.shape[0], spatial_rec.shape[1], f)

                im_spatial_pinv.save(path_save_reconstruction + f'spatial_pinv_depth{d}_wlmean{wl_mean}.png')
                im_spatial_adj.save(path_save_reconstruction + f'spatial_adj_depth{d}_wlmean{wl_mean}.png')
                im_spatial_rec.save(path_save_reconstruction + f'spatial_rec_depth{d}_wlmean{wl_mean}.png')


if __name__ == "__main__":
    name = "performat_letter4.mat"

    # Distances from relay wall at which reconstruct the hidden scene
    depths = [1]
    # Wavelength mean for the virtual illumination pulse in time-frequency domain as described in
    # Phasor Fields
    wl_mean = [0.05]
    # Wavelength standard deviation for the virtual illumination pulse in time-frequency domain
    # as described in Phasor Fields
    # wl_sigma = [] # this is performed automatically as [wl_mean - 0.02, wl_mean - 0.01, wl_mean + 0.01, wl_mean + 0.02]
    # Parameter to regularize the pseudoinverse diffraction operator
    pinv_reg = [0.03, 0.1]

    plot_multi_wavelength(name, depths, wl_mean, pinv_reg)
