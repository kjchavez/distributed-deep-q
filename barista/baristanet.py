import urllib2
import numpy as np
import posix_ipc

import caffe
import barista
from barista.messaging import create_gradient_message
from barista.messaging import create_model_message
from barista.messaging import load_model_message
from barista.ipc_utils import create_shmem_ndarray


class BaristaNet:
    def __init__(self, architecture, model, driver):
        self.net = caffe.Net(architecture, model)

        # TODO: set extra model parameters?
        self.driver = driver

        def create_caffe_shmem_array(name):
            return create_shmem_ndarray('/'+name,
                                        self.net.blobs[name].data.shape,
                                        np.float32,
                                        flags=posix_ipc.O_CREAT)

        # Allocate shared memory for all memory data layers in the network
        self.shared_arrays = {}
        self.shmem = {}
        self._null_array = None
        self.batch_size = None
        for idx, (name, layer) in enumerate(zip(self.net._layer_names, self.net.layers)):
            if layer.type != barista.MEMORY_DATA_LAYER:
                continue

            print "Layer '%s' is a MemoryDataLayer" % name
            handles = name.split('-', 1)

            for name in handles:
                print "Allocating shared memory for %s." % name
                shmem, arr = create_caffe_shmem_array(name)
                self.shared_arrays[name] = arr
                self.shmem[name] = shmem
                if self.batch_size:
                    assert(arr.shape[0] == self.batch_size)
                else:
                    self.batch_size = arr.shape[0]

            if len(handles) == 2:
                self.net.set_input_arrays(self.shared_arrays[handles[0]],
                                          self.shared_arrays[handles[1]],
                                          idx)
            elif len(handles) == 1:
                if self._null_array is None:
                    shape = (self.shared_arrays[handles[0]].shape[0], 1, 1, 1)
                    self._null_array = np.zeros(shape, dtype=np.float32)
                self.net.set_input_arrays(self.shared_arrays[handles[0]],
                                          self._null_array, idx)

        # Allow convenient access to certain Caffe net properties
        self.blobs = self.net.blobs

        # Create semaphore for interprocess synchronization
        self.compute_semaphore = posix_ipc.Semaphore(None, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL)
        self.model_semaphore = posix_ipc.Semaphore(None, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL)

    def fetch_model(self):
        """ Get model parameters from driver over the network. """
        request = urllib2.Request(
                    'http://%s/api/v1/latest_model' % self.driver,
                    headers={'Content-Type': 'application/deepQ'})

        message = urllib2.urlopen(request).read()
        load_model_message(message, self.net)

    def dummy_fetch_model(self):
        """ Returns a model as if it had been retrieved from network.
        """
        message = create_model_message(self.net)  # pretend we recieve this
        load_model_message(message, self.net)

    def send_gradient_update(self):
        """ Sends message as HTTP request; blocks until response is received.
        Raises URLError if cannot be reach driver.
        """
        message = create_gradient_message(self.net)
        try:
            request = urllib2.Request(
                          'http://%s/api/v1/update_model' % self.driver,
                          headers={'Content-Type': 'application/deepQ'},
                          data=message)
            response = urllib2.urlopen(request).read()
        except urllib2.URLError as e:
            print e.message
            raise

        return response

    def dummy_send_gradient_update(self):
        create_gradient_message(self.net)
        p = np.random.rand()
        if p < 0.98:
            response = "OK"
        else:
            response = "ERROR"

        return response

    def full_pass(self):
        self.compute_semaphore.acquire()
        self.net.forward()
        self.net.backward()
        self.model_semaphore.release()

    def forward(self, end=None):
        self.net.forward(end=end)

    def get_ipc_interface(self):
        interface = []
        for name in self.shared_arrays:
            handle = (self.shmem[name].name,
                      self.shared_arrays[name].shape,
                      str(self.shared_arrays[name].dtype))
            interface.append(handle)

        return (self.compute_semaphore.name,
                self.model_semaphore.name,
                interface)

    def ipc_interface_str(self):
        interface_str = ""
        comp_sem, model_sem, interface = self.get_ipc_interface()
        interface_str += comp_sem + '\n'
        interface_str += model_sem + '\n'
        for name, shape, dtype in interface:
            interface_str += ':'.join([name, str(shape), dtype]) + '\n'

        return interface_str

    def __del__(self):
        for shmem in self.shmem.values():
            shmem.close_fd()
            shmem.unlink()

        self.compute_semaphore.unlink()
        self.compute_semaphore.close()
        self.model_semaphore.unlink()
        self.model_semaphore.close()
