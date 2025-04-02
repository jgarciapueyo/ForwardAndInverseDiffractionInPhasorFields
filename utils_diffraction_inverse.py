import numpy as np


def inverse_diffraction_spatial_form(
    u_z: np.ndarray, g_z: np.ndarray, rcond: float = 0.1
):
    """
    Computes the inverse diffraction operators (Table 1) in spatial domain (Eq. 18, 20, 22) and
     applies them to the output wavefield u_z to recover the input wavefield u_0.

    Parameters
    ----------
    u_z : np.ndarray
        Output wavefield in spatial domain.
    g_z : np.ndarray
        Forward diffraction operator in spatial domain.
    rcond : float
        Parameter to regularize the computation of the pseudoinverse diffraction operator.

    Returns
    -------
    A tuple containing the recovered input wavefield using pseudoinverse, adjoint and reciprocity
    theorem inverse diffraction operators, and also the pseudoinverse diffraction operator in
    spatial domain.
    """
    # Forms of inverse diffraction from the output wavefield in spatial domain
    # 1. Inverse diffraction in space domain with the pseudoineverse (Eq. 18)
    g_z_pinv = np.linalg.pinv(g_z, rcond=rcond, hermitian=False)
    u_0_estimated_pinv = g_z_pinv @ u_z

    # 2. Inverse diffraction in space domain with the adjoint (Eq. 20)
    g_z_adj = np.conj(g_z).T
    u_0_estimated_adj = g_z_adj @ u_z

    # 3. Inverse diffraction in space domain with the reciprocity theorem (Eq. 22)
    u_0_estimated_rec = np.conj(g_z @ np.conj(u_z))

    return u_0_estimated_pinv, u_0_estimated_adj, u_0_estimated_rec, g_z_pinv


def inverse_diffraction_spatialfrequency_forms(U_z, G_z, mask):
    """
    Computes the inverse diffraction operators (Table 1) in spatial-frequency domain (Eq. 19, 21, 23)
     and applies them to the output wavefield U_z to recover the input wavefield U_0.

    Parameters
    ----------
    U_z : np.ndarray
        Output wavefield in spatial-frequency domain
    G_z : np.ndarray
        Forward diffraction operator in spatial-frequency domain
    mask : float
        Mask to identify the spatial-frequencies corresponding to homogeneous waves

    Returns
    -------
    A tuple containing the recovered input wavefield using reciprocal, adjoint and reciprocity
    theorem inverse diffraction operators.
    """
    # Forms of inverse diffraction from the output wavefield in spatial-frequency domain
    # 1. Inverse diffraction in spatial-frequency domain with the reciprocal (Eq. 17)
    G_z_reciprocal = np.zeros(G_z.shape, dtype=np.cdouble)
    G_z_reciprocal[mask] = np.reciprocal(G_z)[mask]
    U_0_estimated_reciprocal = np.multiply(G_z_reciprocal, U_z)

    # 2. Inverse diffraction in spatial-frequency domain with the adjoint (Eq. 19)
    G_z_adj = np.conj(G_z)
    U_0_estimated_adj = np.multiply(G_z_adj, U_z)

    # 3. Inverse diffraction in spatial-frequency domain with the reciprocity theorem (Eq. 21)
    U_0_estimated_rec = np.conj(np.multiply(G_z, np.conj(U_z)))

    return U_0_estimated_reciprocal, U_0_estimated_adj, U_0_estimated_rec
