import numpy as np
import numpy.typing as npt
from dolfinx.cpp.graph import AdjacencyList_int32  # type: ignore


def reshape_map(connect: AdjacencyList_int32) -> npt.NDArray[np.int32]:
    """Reshape the connected entities mapping.

    Args:
        connect: the connectivity.

    Returns:
        The mapping as a ndarray.
    """
    array = connect.array
    num_e1_per_e2 = np.diff(connect.offsets)
    max_offset = num_e1_per_e2.max()
    emap = -np.ones((len(connect.offsets) - 1, max_offset), dtype=int)

    # Mask to select the boundary facets
    for num in np.unique(num_e1_per_e2):
        mask = np.where(num_e1_per_e2 == num)[0]
        for n in range(num):
            emap[mask, n] = array[num_e1_per_e2.cumsum()[mask] - n - 1]
    return emap, max_offset
