import urllib2
import numpy as np
import caffe
import barista
from barista.messaging import create_gradient_message
from barista.messaging import create_model_message
from barista.messaging import load_model_message


class BaristaNet:
    def __init__(self, architecture, model, driver, dataset=None):
        self.net = caffe.Net(architecture, model)
        self.dataset = dataset

        # TODO: set extra model parameters?
        self.driver = driver

        assert('state' in self.net.blobs and 'action' in self.net.blobs and
               'reward' in self.net.blobs and 'next_state' in self.net.blobs)

        # Allocate memory for all inputs to the network
        self.state = np.zeros(self.net.blobs['state'].data.shape, dtype=np.float32)
        self.action = np.zeros(self.net.blobs['action'].data.shape, dtype=np.float32)
        self.reward = np.zeros(self.net.blobs['reward'].data.shape, dtype=np.float32)
        self.next_state = np.zeros(self.net.blobs['next_state'].data.shape, dtype=np.float32)
        self.batch_size = self.state.shape[0]

        # Set these as inputs to appropriate IN-MEMORY layers of Caffe
        # TODO: state and next_state layers shouldn't have a "labels" source,
        # but the set_input_arrays function requires two sections of memory
        self.net.set_input_arrays(self.state, self.reward, barista.STATE_MD_LAYER)
        self.net.set_input_arrays(self.next_state, self.reward, barista.NEXT_STATE_MD_LAYER)
        self.net.set_input_arrays(self.action, self.reward, barista.ACTION_REWARD_MD_LAYER)

        # Make sure IN-MEMORY data layers are properly configured
        assert_in_memory_config(self)

    def add_dataset(self, dset):
        self.dataset = dset

    def load_minibatch(self):
        """ Reads a random sample from the replay dataset and writes it Caffe-visible
            memory.
        """
        if self.dataset:
            self.dataset.sample_direct(self.state,
                                       self.action,
                                       self.reward,
                                       self.next_state,
                                       self.batch_size)
        else:
            print "Warning: no dataset specified, using dummy data."
            self.dummy_load_minibatch()

    def dummy_load_minibatch(self):
        """ Writes random data into the numpy arrays.
        """
        self.state[...] = np.random.randint(0, 256, size=self.state.shape)

        # Actions matrix has a one-hot representation
        self.action[...] = np.zeros(self.action.shape)
        self.action[np.arange(self.action.shape[0]),
                    np.random.randint(0, self.action.shape[1],
                                      size=(self.action.shape[0],))] = 1

        self.reward[...] = np.random.randint(-5, 6, size=self.reward.shape)
        self.next_state[...] = np.random.randint(0, 256, size=self.next_state.shape)

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
        message = create_gradient_message(self.net)
        p = np.random.rand()
        if p < 0.98:
            response = "OK"
        else:
            response = "ERROR"

        return response

    def full_pass(self):
        self.net.forward()
        self.net.backward()

    def select_action(self, state):
        self.state[0] = state
        self.net.forward(end='Q_out')
        action = np.argmax(self.net.blobs['Q_out'].data[0], axis=0).squeeze()
        return action


# Auxiliary functions
def assert_in_memory_config(barista_net):
    print "-" * 50
    print "Checking IN-MEMORY data layer configuration..."
    net = barista_net.net
    barista_net.dummy_load_minibatch()

    # Nothing should be loaded in data blobs before calling net forward
    assert(not np.all(net.blobs['state'].data == barista_net.state))

    net.forward()
    assert(np.all(net.blobs['state'].data == barista_net.state))
    assert(np.all(net.blobs['next_state'].data == barista_net.next_state))
    assert(np.all(net.blobs['action'].data == barista_net.action))
    assert(np.all(net.blobs['reward'].data == barista_net.reward))

    # Should read from the IN-MEMORY location directly, no copy
    pointer_local, _ = barista_net.state.__array_interface__['data']
    pointer_caffe, _ = net.blobs['state'].data.__array_interface__['data']
    print "state data mem address (local):", hex(pointer_local)
    print "state data mem address (caffe):", hex(pointer_caffe)
    assert(pointer_local == pointer_caffe)

    pointer_local, _ = barista_net.next_state.__array_interface__['data']
    pointer_caffe, _ = net.blobs['next_state'].data.__array_interface__['data']
    print "next_state data mem address (local):", hex(pointer_local)
    print "next_state data mem address (caffe):", hex(pointer_caffe)
    assert(pointer_local == pointer_caffe)

    print "IN-MEMORY data layers correctly configured."


def test_action_selection():
    baristanet = BaristaNet('models/deepq/train_val.prototxt',
                            'models/deepq/deepq.caffemodel',
                            'Augustus')

    for _ in xrange(10):
        state = np.random.rand(4, 128, 128)
        opt_action = baristanet.select_action(state)
        print opt_action

if __name__ == "__main__":
    test_action_selection()
