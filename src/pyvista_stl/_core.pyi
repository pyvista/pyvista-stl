import numpy as np
import numpy.typing as npt

def get_stl_data(
    filename: str, threads: int = 1
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]: ...
