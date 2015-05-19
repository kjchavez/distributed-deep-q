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
from barista import netutils
from replay import ReplayDataset
from gamesim.SnakeGame import SnakeGame, gray_scale
from expgain import ExpGain, generate_preprocessor

# Modules necessary only for faking Experience Gainer
import random
import numpy as np


def recv_all(socket, size):
    message = ""
    while len(message) < size:
        chunk = socket.recv(4096)
        if not chunk:
            break
        message += chunk

    return message


def process_connection(socket, net, exp_gain, iter_num=1, log_frequency=50):
    print "Processing...",
    message = recv_all(socket, barista.MSG_LENGTH)
    if message == barista.GRAD_UPDATE:
        exp_gain.generate_experience(iter_num)
        net.fetch_model()
        net.load_minibatch()
        net.full_pass()
        response = net.send_gradient_update()
        if iter_num % log_frequency == 0:
            net.log()
        socket.send(response)

    elif message == barista.DARWIN_UPDATE:
        raise NotImplementedError("Cannot process request " + message +
                                  "; Darwinian SGD not implemented")

    else:
        print "Unknown request:", message

    socket.close()
    print "done."


def debug_process_connection(socket, net, exp_gain, iter_num=1):
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
        print "    * Caffe took % 0.2f milliseconds." % (1000 * (toc - tic))

        # Compute debug info
        param_norms = netutils.compute_param_norms(net.net)
        grad_norms = netutils.compute_gradient_norms(net.net)
        loss = netutils.extract_net_data(net.net, ('loss',))['loss']

        print
        print "Parameter norms:"
        print "-"*50
        netutils.pretty_print(param_norms)
        print
        print "Gradient norms:"
        print "-"*50
        netutils.pretty_print(grad_norms)
        print
        print "Loss:", loss

        print "- Generating/sending gradient message..."
        response = net.send_gradient_update()
        net.log()
        socket.send(response)

    elif message == barista.DARWIN_UPDATE:
        raise NotImplementedError("Cannot process request " + message +
                                  "; Darwinian SGD not implemented")

    else:
        print "Unknown request:", message

    socket.close()
    print "Closed connection"

def issue_ready_signal(idx):
    if not os.path.isdir("flags"):
        os.makedirs("flags")

    with open('flags/__BARISTA_READY__.%d' % idx, 'w') as fp:
        pass


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
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.driver == "None":
        args.driver = None

    return args


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
    net = BaristaNet(args.architecture, args.model, args.driver,
                     reset_log=True)

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
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind(('127.0.0.1', args.port))
    serversocket.listen(5)

    print
    print "*"*80
    print "* Starting BARISTA server: listening on port %d." % args.port
    print "*"*80
    # Signal Spark Executor that Barista is ready to receive connections
    issue_ready_signal(args.port)
    while True:
        (clientsocket, address) = serversocket.accept()
        if args.debug:
            handler = debug_process_connection
        else:
            handler = process_connection

        client_thread = threading.Thread(
                            target=handler,
                            args=(clientsocket, net, exp_gain))
        client_thread.run()

if __name__ == "__main__":
    main()
