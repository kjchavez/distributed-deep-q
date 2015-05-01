"""
Barista serves as an interface to a long-running caffe process.
"""
import os
import sys
import time
import argparse
import socket
import threading

import caffe
from caffe import SGDSolver

import barista
from barista.baristanet import BaristaNet
from replay import ReplayDataset

# Modules necessary only for faking Experience Gainer
import random
import numpy as np

def process_connection(socket, net):
    message = ""
    while len(message) < barista.MSG_LENGTH:
        chunk = socket.recv(4096)
        if not chunk:
            break
        message += chunk

    if message == barista.GRAD_UPDATE:
        print "Processing gradient update request:"
        print "- Fetching model..."
        net.fetch_model()
        print net.net.params['Qconv1'][0].data
        print net.net.params['Qconv1'][0].diff
        print np.linalg.norm(net.net.params['Qconv1'][0].data -  
                             net.net.params['Qconv1'][0].diff)
        assert(np.all(net.net.params['Qconv1'][0].data == 
               net.net.params['Qconv1'][0].diff))
        print "- Loading minibatch..."
        net.load_minibatch()
        print "- Running Caffe..."
        tic = time.time()
        net.full_pass()
        toc = time.time()
        print "Caffe took % 0.2f milliseconds." % (1000 * (toc - tic))
        print "- Generating/sending gradient message..."
        response = net.send_gradient_update()
        socket.send(response)

    elif message == barista.DARWIN_UPDATE:
        raise NotImplementedError("Cannot process request " + message +
                                  "; Darwinian SGD not implemented")

    else:
        print "Unknown request:", message

    socket.close()
    print "Closed connection"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("model")
    parser.add_argument("--solver", default=None)
    parser.add_argument("--mode", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--driver", default="127.0.0.1:5500")
    parser.add_argument("--dataset", default="replay-dataset.hdf5")
    parser.add_argument("--dset-size", dest="dset_size", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    caffe.set_phase_test()
    if args.mode == "cpu":
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    if not os.path.isfile(args.model):
        if not args.solver:
            print "Error: Model does not exist. No solver specified."
            sys.exit(1)

        print "Warning: model %s does not exist. Creating..."
        solver = SGDSolver(args.solver)
        solver.net.save(args.model)

    # Initialize objects
    net = BaristaNet(args.architecture, args.model, args.driver)
    replay_dataset = ReplayDataset(args.dataset, net.state[0].shape,
                         dset_size=args.dset_size, overwrite=args.overwrite)
    net.add_dataset(replay_dataset)

    # TODO: Fill the replay dataset with real experiences
    print "Filling replay dataset with random experiences..."
    for _ in xrange(100):
        action = random.choice(xrange(4))
        reward = random.choice(xrange(-5, 6))
        state = np.random.randint(0, 256, size=net.state[0].shape)
        replay_dataset.add_experience(action, reward, state)
    print "Done."
    net.send_gradient_update()

    # Start server loop
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('127.0.0.1', args.port))
    serversocket.listen(5)

    print
    print "*"*80
    print "* Starting BARISTA server: listening on port %d." % args.port
    print "*"*80
    while True:
        (clientsocket, address) = serversocket.accept()
        print "Accepted connection"
        client_thread = threading.Thread(target=process_connection,
                                         args=(clientsocket, net))
        client_thread.run()

if __name__ == "__main__":
    main()
