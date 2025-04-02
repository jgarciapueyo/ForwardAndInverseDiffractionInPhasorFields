from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
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
    im = Image.fromarray(np.uint8(cmap(normalised) * 255))
    return im


def plot_wavelength_distance_vs_rank(
    name: str,
    depths: List[float],
    wl_mean_list: List[float],
    pr: float,
    asize: List[float],
    sp: int,
):
    """
    Creates a plot to show how the relative rank of the forward diffraction operator g_z
    in matrix form evolves for different depths D for different wavelengths.
    This code generates Fig. 7a of the paper.

    Parameters
    ----------
    name : str
        Path to the data to load.
    depths : List[float]
        List of depths at which the hidden object has been reconstructed.
    wl_mean_list : List[float]
        Wavelengths of the virtual illumination pulse (in time-frequency domain)
        as defined in Phasor Fields
    pr : float
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    asize : float
        Size (in m) of the relay wall.
    sp : int
        Number of sampling points in the relay wall.
    """
    filepath = f"results/{name}/"
    rank_per_wl = []
    condition_number_per_wl = []
    reconstructions = []

    for wl_mean in wl_mean_list:
        rank_at_depth_per_wl = []
        condition_number_at_depth_per_wl = []
        reconstructions_per_wl = []

        for d in depths:
            data = np.load(
                filepath
                + f"R_{d}m_{sp}sp_a{asize}m/R_{d}m_{sp}sp_a{asize}m_{wl_mean}wl_{pr}pinvreg.npy",
                allow_pickle=True,
            ).item()
            rank_at_depth_per_wl.append(data["rank"] / (sp**2))
            condition_number_at_depth_per_wl.append(data["condition_number"])
            reconstructions_per_wl.append(data["reconstructions"])

        rank_per_wl.append(rank_at_depth_per_wl)
        condition_number_per_wl.append(condition_number_at_depth_per_wl)
        reconstructions.append(reconstructions_per_wl)

    plt.figure(1, figsize=(5.5, 5.5))
    new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
    mpl.rcParams.update(new_rc_params)

    for i_wl, wl_mean in enumerate(wl_mean_list):
        plt.plot(depths, rank_per_wl[i_wl], "o--", label=f"{wl_mean:.2f} m")

    plt.xticks(depths)
    plt.yticks(np.arange(0, 1.05, step=0.1))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x} m"))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlabel("Distance ($D$)")
    plt.ylabel("Relative rank ($r$)")
    plt.grid(color=(0.8, 0.8, 0.8))
    plt.legend(title="Virtual\nWavelength ($\lambda_v$)", loc="upper right")
    plt.tight_layout()
    plt.savefig(
        f"results/{name}_plot/wavelength_distance_vs_rank/{sp}sp_a{asize}m_{pr}pinvreg_plot.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.figure(2, figsize=(20, 10))
    i = 1
    f = lambda x: np.abs(x)
    for i_wl, wl in enumerate(wl_mean_list):
        for i_d, d in enumerate(depths):
            plt.subplot(len(wl_mean_list), len(depths), i)
            r = rank_per_wl[i_wl][i_d]
            rc = 1.22 * d * wl / (asize)
            plt.gca().set_title(f"r={round(r, 3)}\nrc={round(rc, 3)}")
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(f(reconstructions[i_wl][i_d]["spatial_pinv"][0]), cmap="hot")
            i += 1
    plt.savefig(
        f"results/{name}_plot/wavelength_distance_vs_rank/{sp}sp_a{asize}m_{pr}pinvreg_reconstructions.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.figure(3, figsize=(8, 8))
    i = 1
    f = lambda x: np.abs(x)
    for i_wl, wl in enumerate(wl_mean_list):
        for i_d, d in enumerate(depths):
            if i_d % 2 == 0:
                continue
            plt.subplot(len(wl_mean_list), len(depths) // 2, i)
            r = rank_per_wl[i_wl][i_d]
            rc = 1.22 * d * wl / (asize)
            plt.gca().set_title(f"r={round(r, 3)}\nrc={round(rc, 3)}")
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(f(reconstructions[i_wl][i_d]["spatial_pinv"][0]), cmap="hot")
            i += 1
    plt.savefig(
        f"results/{name}_plot/wavelength_distance_vs_rank/{sp}sp_a{asize}m_{pr}pinvreg_reconstructions_simple.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def reconstructions_wavelength_distance_vs_rank(
    name: str,
    depths: List[float],
    wl_mean_list: List[float],
    pr: float,
    asize: List[float],
    sp: int,
):
    """
    Saves the images of the reconstructions done when executing `computational_metric.py`.
    This code generates Fig. 8a of the paper.

    Parameters
    ----------
    name : str
        Path to the data to load.
    depths : List[float]
        List of depths at which the hidden object has been reconstructed.
    wl_mean_list : List[float]
        Wavelengths of the virtual illumination pulse (in time-frequency domain)
        as defined in Phasor Fields
    pr : float
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    asize : float
        Size (in m) of the relay wall.
    sp : int
        Number of sampling points in the relay wall.
    """
    filepath = f"results/{name}/"
    rank_per_wl = []
    condition_number_per_wl = []
    reconstructions = []

    for wl_mean in wl_mean_list:
        rank_at_depth_per_wl = []
        condition_number_at_depth_per_wl = []
        reconstructions_per_wl = []

        for d in depths:
            data = np.load(
                filepath
                + f"R_{d}m_{sp}sp_a{asize}m/R_{d}m_{sp}sp_a{asize}m_{wl_mean}wl_{pr}pinvreg.npy",
                allow_pickle=True,
            ).item()
            rank_at_depth_per_wl.append(data["rank"] / (sp**2))
            condition_number_at_depth_per_wl.append(data["condition_number"])
            reconstructions_per_wl.append(data["reconstructions"])

        rank_per_wl.append(rank_at_depth_per_wl)
        condition_number_per_wl.append(condition_number_at_depth_per_wl)
        reconstructions.append(reconstructions_per_wl)

    Path(
        f"results/{name}_plot/wavelength_distance_vs_rank/{sp}sp_a{asize}m_{pr}pinvreg"
    ).mkdir(exist_ok=True, parents=True)

    f = lambda x: np.abs(x)
    for i_wl, wl in enumerate(wl_mean_list):
        for i_d, d in enumerate(depths):
            reconstruction = reconstructions[i_wl][i_d]["spatial_pinv"][0]
            im = fromarray(
                reconstruction, reconstruction.shape[0], reconstruction.shape[1], f
            )
            im.save(
                f"results/{name}_plot/wavelength_distance_vs_rank/{sp}sp_a{asize}m_{pr}pinvreg/{d}m_{wl}wl.png"
            )


def plot_aperturesize_distance_vs_rank(
    name: str,
    depths: List[float],
    wl_mean: float,
    pr: float,
    asize_list: List[float],
    sp: int,
):
    """
    Creates a plot to show how the relative rank of the forward diffraction operator g_z
    in matrix form evolves for different depths D for aperture sizes A.
    This code generates Fig. 7b of the paper.

    Parameters
    ----------
    name : str
        Path to the data to load.
    depths : List[float]
        List of depths at which the hidden object has been reconstructed.
    wl_mean : float
        Wavelength of the virtual illumination pulse (in time-frequency domain)
        as defined in Phasor Fields
    pr : float
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    asize_list : List[float]
        List of size (in m) of the relay wall.
    sp : int
        Number of sampling points in the relay wall.
    """
    filepath = f"results/{name}/"
    rank_per_asize = []
    condition_number_per_asize = []
    reconstructions = []

    for asize in asize_list:
        rank_at_depth_per_asize = []
        condition_number_at_depth_per_asize = []
        reconstructions_per_asize = []

        for d in depths:
            data = np.load(
                filepath
                + f"R_{d}m_{sp}sp_a{asize}m/R_{d}m_{sp}sp_a{asize}m_{wl_mean}wl_{pr}pinvreg.npy",
                allow_pickle=True,
            ).item()
            rank_at_depth_per_asize.append(data["rank"] / (sp**2))
            condition_number_at_depth_per_asize.append(data["condition_number"])
            reconstructions_per_asize.append(data["reconstructions"])

        rank_per_asize.append(rank_at_depth_per_asize)
        condition_number_per_asize.append(condition_number_at_depth_per_asize)
        reconstructions.append(reconstructions_per_asize)

    plt.figure(1, figsize=(5.5, 5.5))
    for i_asize, asize in enumerate(asize_list):
        plt.plot(depths, rank_per_asize[i_asize], "o--", label=f"{asize} m")

    plt.xticks(depths)
    plt.yticks(np.arange(0, 1.05, step=0.1))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x} m"))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlabel("Distance ($D$)")
    plt.ylabel("Relative rank ($r$)")
    plt.grid(color=(0.8, 0.8, 0.8))
    plt.legend(title="Aperture\nSize ($\mathcal{A}$)", loc="upper right")
    plt.tight_layout()
    plt.savefig(
        f"results/{name}_plot/aperturesize_distance_vs_rank/{wl_mean}wl_{sp}sp_{pr}pinvreg_plot.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.figure(2, figsize=(10, 8))
    i = 1
    f = lambda x: np.abs(x)
    for i_asize, asize in enumerate(asize_list):
        for i_d, d in enumerate(depths):
            plt.subplot(len(asize_list), len(depths), i)
            plt.title(
                f"r={round(rank_per_asize[i_asize][i_d], 3)}\nrc={round(1.22*d*wl_mean/(asize),3)}"
            )
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(f(reconstructions[i_asize][i_d]["spatial_pinv"][0]), cmap="hot")
            i += 1
    plt.savefig(
        f"results/{name}_plot/aperturesize_distance_vs_rank/{wl_mean}wl_{sp}sp_{pr}pinvreg_reconstructions.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.figure(3, figsize=(10, 8))
    i = 1
    f = lambda x: np.abs(x)
    for i_asize, asize in enumerate(asize_list):
        for i_d, d in enumerate(depths):
            if i_d % 2 == 1:
                continue
            plt.subplot(len(asize_list), len(depths) // 2, i)
            plt.title(
                f"r={round(rank_per_asize[i_asize][i_d], 3)}\nrc={round(1.22*d*wl_mean/(asize),3)}"
            )
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(f(reconstructions[i_asize][i_d]["spatial_pinv"][0]), cmap="hot")
            i += 1
    plt.savefig(
        f"results/{name}_plot/aperturesize_distance_vs_rank/{wl_mean}wl_{sp}sp_{pr}pinvreg_reconstructions_simple.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    # plt.show()


def reconstructions_aperturesize_distance_vs_rank(
    name: str,
    depths: List[float],
    wl_mean: float,
    pr: float,
    asize_list: List[float],
    sp: int,
):
    """
    Saves the images of the reconstructions done when executing `computational_metric.py`.
    This code generates Fig. 9a of the paper.

    Parameters
    ----------
    name : str
        Path to the data to load.
    depths : List[float]
        List of depths at which the hidden object has been reconstructed.
    wl_mean : float
        Wavelength of the virtual illumination pulse (in time-frequency domain)
        as defined in Phasor Fields
    pr : float
        threshold used to compute the pseudoinverse diffraction operator (t_th in the paper)
    asize_list : List[float]
        List of size (in m) of the relay wall.
    sp : int
        Number of sampling points in the relay wall.
    """
    filepath = f"results/{name}/"
    rank_per_asize = []
    condition_number_per_asize = []
    reconstructions = []

    for asize in asize_list:
        rank_at_depth_per_asize = []
        condition_number_at_depth_per_asize = []
        reconstructions_per_asize = []

        for d in depths:
            data = np.load(
                filepath
                + f"R_{d}m_{sp}sp_a{asize}m/R_{d}m_{sp}sp_a{asize}m_{wl_mean}wl_{pr}pinvreg.npy",
                allow_pickle=True,
            ).item()
            rank_at_depth_per_asize.append(data["rank"] / (sp**2))
            condition_number_at_depth_per_asize.append(data["condition_number"])
            reconstructions_per_asize.append(data["reconstructions"])

        rank_per_asize.append(rank_at_depth_per_asize)
        condition_number_per_asize.append(condition_number_at_depth_per_asize)
        reconstructions.append(reconstructions_per_asize)

    Path(
        f"results/{name}_plot/aperturesize_distance_vs_rank/{sp}sp_{wl_mean}wl_{pr}pinvreg"
    ).mkdir(exist_ok=True, parents=True)

    f = lambda x: np.abs(x)
    for i_asize, asize in enumerate(asize_list):
        for i_d, d in enumerate(depths):
            reconstruction = reconstructions[i_asize][i_d]["spatial_pinv"][0]
            im = fromarray(
                reconstruction, reconstruction.shape[0], reconstruction.shape[1], f
            )
            im.save(
                f"results/{name}_plot/aperturesize_distance_vs_rank/{sp}sp_{wl_mean}wl_{pr}pinvreg/{d}m_a{asize}m.png"
            )


if __name__ == "__main__":
    # PLOT SIMILAR TO Fig. 7a and Fig. 8a
    name = "R"
    # Distances z of the hidden object letter R from the relay wall
    # These are also the distances D at which reconstruct the hidden object
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Wavelength mean for the virtual illumination pulse in time-frequency domain as described in
    # Phasor Fields
    wl_mean = [0.01, 0.03, 0.06, 0.10, 0.13]
    # Parameter to regularize the pseudoinverse diffraction operator
    pinv_reg = [0.1]
    # Size of the relay wall A (similar to size of the aperture)
    aperture_size = [2, 3, 4]
    # Number of scanned points in the relay wall N_s
    sampling_points = [32]
    plot_wavelength_distance_vs_rank(
        name, depths, wl_mean, pinv_reg[0], aperture_size[0], sampling_points[0]
    )
    reconstructions_wavelength_distance_vs_rank(
        name, depths, wl_mean, pinv_reg[0], aperture_size[0], sampling_points[0]
    )

    # PLOT SIMILAR TO Fig. 7b and Fig. 9a
    name = "R"
    # Distances z of the hidden object letter R from the relay wall
    # These are also the distances D at which reconstruct the hidden object
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Wavelength mean for the virtual illumination pulse in time-frequency domain as described in
    # Phasor Fields
    wl_mean = [0.10]
    # Parameter to regularize the pseudoinverse diffraction operator
    pinv_reg = [0.1]
    # Size of the relay wall A (similar to size of the aperture)
    aperture_size = [2, 3, 4]
    # Number of scanned points in the relay wall N_s
    sampling_points = [32]
    reconstructions_aperturesize_distance_vs_rank(
        name, depths, wl_mean[0], pinv_reg[0], aperture_size, sampling_points[0]
    )
    plot_aperturesize_distance_vs_rank(
        name, depths, wl_mean[0], pinv_reg[0], aperture_size, sampling_points[0]
    )
