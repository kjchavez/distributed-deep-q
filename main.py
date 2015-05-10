"""
Barista serves as an interface to a long-running caffe process.
"""
import os
import sys
import time
import argparse
import socket
import threading
import urllib2

import caffe
from caffe import SGDSolver

import barista
from barista.baristanet import BaristaNet
from replay import ReplayDataset
from gamesim.SnakeGame import SnakeGame, gray_scale
from ExpGain import ExpGain, generate_preprocessor

# Modules necessary only for faking Experience Gainer
import random
import numpy as np


def process_connection(socket, net, exp_gain, iter_num=1):
    message = ""
    while len(message) < barista.MSG_LENGTH:
        chunk = socket.recv(4096)
        if not chunk:
            break
        message += chunk

    if message == barista.GRAD_UPDATE:
        exp_gain.generate_experience(iter_num)
        print "Processing gradient update request:"
        print "- Fetching model..."
        net.fetch_model()
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
                                   dset_size=args.dset_size,
                                   overwrite=args.overwrite)
    net.add_dataset(replay_dataset)

    game = SnakeGame()
    preprocessor = generate_preprocessor(net.state.shape[2:], gray_scale)
    exp_gain = ExpGain(net, ['w', 'a', 's', 'd'], preprocessor, game.cpu_play,
                       replay_dataset, game.encode_state())

    for _ in xrange(50):
        exp_gain.generate_experience(1e8)

    # Start server loop
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('127.0.0.1', args.port))
    serversocket.listen(5)

    print
    print "*"*80
    print "* Starting BARISTA server: listening on port %d." % args.port
    print "*"*80
    with open('flags/__BARISTA_READY__', 'w') as fp:
        pass

    while True:
        (clientsocket, address) = serversocket.accept()
        print "Accepted connection"
        client_thread = threading.Thread(target=process_connection,
                                         args=(clientsocket, net, exp_gain))
        client_thread.run()

if __name__ == "__main__":
    main()
