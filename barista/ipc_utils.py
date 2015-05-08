import posix_ipc
import mmap
import numpy as np


def create_shmem_ndarray(name, shape, dtype, flags=0):
    """ Allocates region of shared memory and wraps with a numpy array.

    Args:
        name: for shared memory object, must start with a / and be a valid
              path name (not necessarily an existing path though, since no
              file is actually created)
        shape: dimensions of numpy array to produce
        dtype: data type for numpy array
        flags: for allocating SharedMemory; see posix_ipc docs

    Returns:
        (SharedMemory, ndarray)
    """
    size = np.prod(shape) * np.dtype(dtype).itemsize
    shmem = posix_ipc.SharedMemory(name,
                                   flags=flags,
                                   size=size)
    buf = mmap.mmap(shmem.fd, shmem.size)
    arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
    return shmem, arr
