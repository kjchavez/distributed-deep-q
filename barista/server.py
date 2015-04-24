"""
Barista serves as an interface to a long-running caffe process.
"""
import os
import time
import argparse
import socket
import threading

import caffe
from caffe import SGDSolver

import barista
from barista.baristanet import BaristaNet


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
        net.dummy_fetch_model()
        print "- Loading minibatch..."
        net.dummy_load_minibatch()
        print "- Running Caffe..."
        tic = time.time()
        net.full_pass()
        toc = time.time()
        print "Caffe took % 0.2f milliseconds." % (1000 * (toc - tic))
        print "- Generating/sending gradient message..."
        response = net.dummy_send_gradient_update()
        print "done."
        socket.send(response)

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
    parser.add_argument("--mode", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--driver", default="127.0.0.1:5000")
    args = parser.parse_args()

    print "PID:", os.getpid()
    caffe.set_phase_test()

    if args.mode == "cpu":
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    if not os.path.isfile(args.model):
        print "Warning: model %s does not exist. Creating..."
        solver = SGDSolver(args.solver)
        solver.net.save(args.model)

    net = BaristaNet(args.architecture, args.model, args.driver)

    # Start server loop
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('127.0.0.1', args.port))
    serversocket.listen(5)

    print "Starting barista server, listening on port %d." % args.port
    while True:
        (clientsocket, address) = serversocket.accept()
        print "Accepted connection"
        client_thread = threading.Thread(target=process_connection,
                                         args=(clientsocket, net))
        client_thread.run()

if __name__ == "__main__":
    main()
