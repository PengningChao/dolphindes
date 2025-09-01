"""Utilities for defining optimization geometries."""

import numpy as np
from numpy.typing import NDArray


def embed_vector(
    des_mask: NDArray[np.bool_], vector_r: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Embeds a distribution vector(r) into the full domain.

    Parameters
    ----------
    des_mask : NDArray[np.bool_]
        2D Boolean mask defining the design region.
    vector_r : NDArray[np.floating]
        1D array of vector values defined only on the design region.
    """
    Nx, Ny = des_mask.shape
    vector_full = np.zeros((Nx, Ny), dtype=vector_r.dtype)
    vector_full[des_mask] = vector_r
    return vector_full
