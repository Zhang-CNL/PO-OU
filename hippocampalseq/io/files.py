import os
import mat73 
import scipy.io as sio


def read_mat(file: str) -> dict[str, Any]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found, make sure you have the complete dataset.")
    try:
        return sio.loadmat(file, squeeze_me=True, struct_as_record=False)
    except:
        return mat73.loadmat(file)
