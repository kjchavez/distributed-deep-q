import posix_ipc
from ipc_utils import create_shmem_ndarray


class Customer(object):
    def __init__(self, compute_semaphore, model_semaphore, handles):
        self.arrays = {}
        self.shmem = {}
        for name, shape, dtype in handles:
            shmem, arr = create_shmem_ndarray(name, shape, dtype, flags=0)
            self.arrays[name[1:]] = arr
            self.shmem[name[1:]] = shmem

        self.compute_semaphore = posix_ipc.Semaphore(compute_semaphore, flags=0)
        self.model_semaphore = posix_ipc.Semaphore(model_semaphore, flags=0)

    def update_data(self):
        raise NotImplementedError("Override this function.")

    def process_model(self):
        raise NotImplementedError("Override this function")

    def run_transaction(self, timeout=None):
        self.update_data()
        self.compute_semaphore.release()
        self.model_semaphore.acquire(timeout)
        self.process_model()

    def __getattr__(self, attr):
        return self.arrays[attr]

    def __del__(self):
        for shmem in self.shmem.values():
            shmem.close()

        self.compute_semaphore.close()
        self.model_semaphore.close()
