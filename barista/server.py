"""
Barista serves as an interface to a long-running caffe process.
"""
import os
import time
import argparse
import socket
import urllib2
import threading
import cPickle

import numpy as np
import caffe
from caffe import SGDSolver

import barista
from barista.messaging import create_gradient_message
from barista.messaging import create_model_message
from barista.messaging import load_model_message


def load_model(arch_file, model_file, image_dims,
               mean_image_file=None, channel_swap=(0, 1, 2)):
    net = caffe.Net(arch_file, model_file)

    # TODO: Set various extra parameters of the network
    return net


def dummy_load_minibatch(states, actions, rewards, next_states):
    """ Writes random data into the numpy arrays.
    """
    batch_size = states.shape[0]
    assert(actions.shape[0] == batch_size)
    assert(rewards.size == batch_size)
    assert(next_states.shape == states.shape)

    states[...] = np.random.randint(0, 256, size=states.shape)

    # Actions matrix has a one-hot representation
    actions[...] = np.zeros(actions.shape)
    actions[:, np.random.randint(0, actions.shape[1])] = 1

    rewards[...] = np.random.randint(-5, 6, size=rewards.shape)
    next_states[...] = np.random.randint(0, 256, size=next_states.shape)


def dummy_fetch_model(net, driver=None):
    """ Returns a model as if it had been retrieved from network.
    """
    message = create_model_message(net)  # pretend we recieve this
    load_model_message(message, net)


def load_minibatch(states, actions, rewards, next_states):
    raise NotImplementedError("Not implemented yet")


def fetch_model(net, driver="localhost:5000"):
    """ Get model parameters from driver over the network. """
    request = urllib2.Request('http://%s/model' % driver,
                              headers={'Content-Type': 'application/deepQ'})
    message = cPickle.loads(urllib2.urlopen(request).read())
    load_model_message(message, net)


def assert_in_memory_config(net, state, action, reward, next_state):
    print "-" * 50
    print "Checking IN-MEMORY data layer configuration..."
    dummy_load_minibatch(state, action, reward, next_state)

    # Nothing should be loaded in data blobs before calling net forward
    assert(not np.all(net.blobs['state'].data == state))

    net.forward()
    assert(np.all(net.blobs['state'].data == state))
    assert(np.all(net.blobs['next_state'].data == next_state))
    assert(np.all(net.blobs['action'].data == action))
    assert(np.all(net.blobs['reward'].data == reward))

    # Should read from the IN-MEMORY location directly, no copy
    pointer_local, _ = state.__array_interface__['data']
    pointer_caffe, _ = net.blobs['state'].data.__array_interface__['data']
    print "state data mem address (local):", hex(pointer_local)
    print "state data mem address (caffe):", hex(pointer_caffe)
    assert(pointer_local == pointer_caffe)

    pointer_local, _ = next_state.__array_interface__['data']
    pointer_caffe, _ = net.blobs['next_state'].data.__array_interface__['data']
    print "next_state data mem address (local):", hex(pointer_local)
    print "next_state data mem address (caffe):", hex(pointer_caffe)
    assert(pointer_local == pointer_caffe)

    print "IN-MEMORY data layers correctly configured."


def send_update(message, dest):
    """ Sends message as HTTP request; blocks until response is received.
    """
    request = urllib2.Request('http://%s/update' % dest,
                              headers={'Content-Type': 'application/deepQ'},
                              data=message)

    return urllib2.urlopen(request).read()


def dummy_send_update(message, dest):
    p = np.random.rand()
    if p < 0.98:
        response = "OK"
    else:
        response = "ERROR"

    return response


def process_connection(socket, driver, net, state, action, reward, next_state):
    message = ""
    while len(message) < barista.MSG_LENGTH:
        chunk = socket.recv(4096)
        if not chunk:
            break
        message += chunk

    if message == barista.GRAD_UPDATE:
        print "Processing gradient update request:"
        print "- Fetching model..."
        dummy_fetch_model(net, driver=driver)
        print "- Loading minibatch..."
        dummy_load_minibatch(state, action, reward, next_state)
        print "- Running Caffe..."
        tic = time.time()
        net.forward()
        net.backward()
        toc = time.time()
        print "Caffe took % 0.2f milliseconds." % (1000 * (toc - tic))
        print "- Generating gradient message..."
        grad_update = create_gradient_message(net)
        print "- Sending..."
        response = dummy_send_update(grad_update, driver)
        socket.send(response)
        print "done."

    elif message == barista.DARWIN_UPDATE:
        raise NotImplementedError("Cannot process request " + message +
                                  "; Darwinian SGD not implemented")

    else:
        print "Unknown request:", message

    socket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("model")
    parser.add_argument("solver")
    parser.add_argument("--image-dims", dest="image_dims",
                        nargs=2, type=int, default=[256, 256])
    parser.add_argument("--mode", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--driver", default="localhost:5000")
    args = parser.parse_args()

    print "PID:", os.getpid()
    caffe.set_phase_test()

    if args.mode == "cpu":
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    if not os.path.isfile("barista/models/deepq/deepq.caffemodel"):
        solver = SGDSolver(args.solver)
        solver.net.save("barista/models/deepq/deepq.caffemodel")

    net = load_model(args.architecture, args.model, args.image_dims)
    assert('state' in net.blobs and 'action' in net.blobs and
           'reward' in net.blobs and 'next_state' in net.blobs)

    # Allocate memory for all inputs to the network
    state = np.zeros(net.blobs['state'].data.shape, dtype=np.float32)
    action = np.zeros(net.blobs['action'].data.shape, dtype=np.float32)
    reward = np.zeros(net.blobs['reward'].data.shape, dtype=np.float32)
    next_state = np.zeros(net.blobs['next_state'].data.shape, dtype=np.float32)

    # Set these as inputs to appropriate IN-MEMORY layers of Caffe
    net.set_input_arrays(state, reward, barista.STATE_MD_LAYER)
    net.set_input_arrays(next_state, reward, barista.NEXT_STATE_MD_LAYER)
    net.set_input_arrays(action, reward, barista.ACTION_REWARD_MD_LAYER)

    # Make sure IN-MEMORY data layers are properly configured
    assert_in_memory_config(net, state, action, reward, next_state)

    # Start server loop
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('127.0.0.1', args.port))
    serversocket.listen(5)

    print "Starting barista server, listening on port %d." % args.port
    while True:
        (clientsocket, address) = serversocket.accept()
        print "Accepted connection"
        client_thread = threading.Thread(target=process_connection,
                                         args=(clientsocket, args.driver,
                                               net, state, action,
                                               reward, next_state))
        client_thread.run()

if __name__ == "__main__":
    main()
