""" Messaging interface between local Caffe process and master node
"""
import sys
import time
import struct
import numpy as np
import cPickle
import zlib
from collections import OrderedDict

import barista


def create_net_message(params, attr, compress=False):
    """Consistent method for generating messages based on Caffe net params.
    Args:
        net: Caffe Net object
        attr: string identifying which attribute of the parameters to use
        compress: if True use zlib compression on data

    Returns:
        A message as a byte string of the form:

                 INT X (4 bytes) | HEADER (X bytes) | DATA
    """
    meta_data = OrderedDict()
    data = ""
    for param in params:
        if param[0] == "Q":  # We only want the parameters for the Q CNN
            meta_data[param] = list(getattr(blob, attr).shape
                                    for blob in params[param])

            # Use Numpy's serialization for the arrays
            for i in xrange(len(params[param])):
                assert(getattr(params[param][i], attr).dtype ==
                       barista.DTYPE)
                data += getattr(params[param][i], attr).tostring()

    header = cPickle.dumps(meta_data, -1)
    if compress:
        data = zlib.compress(data)

    message = struct.pack('i', len(header)) + header + data
    return message


def create_gradient_message(net, compress=False):
    """ Extracts gradients from net's parameters and composes message.
    """
    return create_net_message(net.params, "diff", compress=compress)


def create_model_message(net, compress=False):
    return create_net_message(net.params, "data", compress=compress)


# Functions for loading messages directly into a Caffe net object.
def load_net_message(message, net, attr, compressed=False):
    """ Loads the data received over the network into a net.
        Note: Assumes data is of type float32, but this can be relaxed.
    """
    header_size_message = message[0:4]
    header_size = struct.unpack('i', header_size_message)[0]
    header_message = message[4:4 + header_size]
    header = cPickle.loads(header_message)

    if compressed:
        data = zlib.decompress(message[4 + header_size:])
    else:
        data = message[4 + header_size:]

    idx = 0
    for param in header:
        if param not in net.params:
            raise KeyError("Received parameter %s not in"
                           "model's architecture." % param)

        shapes = header[param]
        lengths = [np.prod(shape)*barista.DTYPE_SIZE for shape in shapes]
        for i, (shape, length) in enumerate(zip(shapes, lengths)):
            getattr(net.params[param][i], attr).flat[:] = \
                np.frombuffer(data[idx:idx+length], dtype=barista.DTYPE)

            idx += length

    if idx != len(data):
        print "Warning: data not entirely consumed (%d / %d bytes used)" \
               % (idx, len(data))


def load_model_message(message, net):
    """ Replace net parameters with those contained in message.
    """
    load_net_message(message, net, "data")


def load_gradient_message(message, compressed=False):
    """ Returns dictionary of parameter name to gradient numpy array.

    Args:
        message: (str) data received from network
        compressed: if True, uses zlib to decompress data

    Return:
        Dictionary of list of gradients for each parameter.
        i.e.
                {"param1": [ndarray, ndarray], "param2": [ndarray]}

        Numpy arrays in this dictionary are immutable.
    """
    header_size_message = message[0:4]
    header_size = struct.unpack('i', header_size_message)[0]
    header_message = message[4:4 + header_size]
    header = cPickle.loads(header_message)

    if compressed:
        data = zlib.decompress(message[4 + header_size:])
    else:
        data = message[4 + header_size:]

    idx = 0
    grads = {}
    for param in header:
        shapes = header[param]
        lengths = [np.prod(shape) for shape in shapes]
        grads[param] = []
        for shape, length in zip(shapes, lengths):
            grads[param].append(
                np.frombuffer(data[idx:idx+length],
                              dtype=barista.DTYPE).reshape(shape))
            idx += length

    return grads


# Evaluation functions
def evaluate_message_generation(net, state, action, reward, next_state):
    tic = time.time()
    net.forward()
    net.backward()
    toc = time.time()
    print "-- MESSAGE INFORMATION --"
    print "Forward/backward pass: %0.3f ms" % (1000 * (toc - tic))

    # Extract gradients
    tic = time.time()
    msg = create_gradient_message(net, compress=False)
    toc = time.time()
    print "Without compression:"
    print "Message size: %0.2f MB" % (sys.getsizeof(msg) / 1.0e6)
    print "Generation time: %0.2f ms" % (1000 * (toc - tic))
    tic = time.time()
    grads = load_gradient_message(msg, compressed=False)
    toc = time.time()
    print "Loading time: %0.2f ms" % (1000 * (toc - tic))

    tic = time.time()
    msg = create_gradient_message(net, compress=True)
    toc = time.time()
    print
    print "With compression:"
    print "Message size: %0.2f MB" % (sys.getsizeof(msg) / 1.0e6)
    print "Generation time: %0.2f ms" % (1000 * (toc - tic))
    tic = time.time()
    grads = load_gradient_message(msg, compressed=True)
    toc = time.time()
    print "Loading time: %0.2f ms" % (1000 * (toc - tic))
    print
    print "Parameters included in message:"
    for param in grads:
        print " -", param


if __name__ == "__main__":
    import caffe

    net = caffe.Net("models/deepq/train_val.prototxt",
                    "models/deepq/deepq.caffemodel")
    assert('state' in net.blobs and 'action' in net.blobs and
           'reward' in net.blobs and 'next_state' in net.blobs)

    # Allocate memory for all inputs to the network
    state = np.random.normal(size=net.blobs['state'].data.shape).astype(np.float32)
    action = np.random.normal(size=net.blobs['action'].data.shape).astype(np.float32)
    reward = np.random.normal(size=net.blobs['reward'].data.shape).astype(np.float32)
    next_state = np.random.normal(size=net.blobs['next_state'].data.shape).astype(np.float32)

    # Set these as inputs to appropriate IN-MEMORY layers of Caffe
    net.set_input_arrays(state, reward, barista.STATE_MD_LAYER)
    net.set_input_arrays(next_state, reward, barista.NEXT_STATE_MD_LAYER)
    net.set_input_arrays(action, reward, barista.ACTION_REWARD_MD_LAYER)

    # Evaluate timing and sizes of generated messages
    evaluate_message_generation(net, state, action, reward, next_state)

    import cPickle
    message = create_gradient_message(net)
    with open("message.pkl",'w') as fp:
        cPickle.dump(message, fp)
