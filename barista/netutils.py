import os
import shutil
import numpy as np
from datetime import datetime


def compute_gradient_norms(caffe_net, params=None, ord=None):
    norms = {}
    if not params:
        params = caffe_net.params

    for param in params:
        norms[param] = []
        for i in range(len(caffe_net.params[param])):
            norm = np.linalg.norm(np.ravel(caffe_net.params[param][i].diff), ord=ord)
            if ord != 0:
                norm /= np.sqrt(caffe_net.params[param][i].diff.size)
            norms[param].append(norm)

    return norms


def compute_param_norms(caffe_net, params=None, ord=None):
    norms = {}
    if not params:
        params = caffe_net.params

    for param in params:
        norms[param] = []
        for i in range(len(caffe_net.params[param])):
            norm = np.linalg.norm(np.ravel(caffe_net.params[param][i].data), ord=ord)
            if ord != 0:
                norm /= np.sqrt(caffe_net.params[param][i].diff.size)
            norms[param].append(norm)

    return norms


def compute_data_norms(caffe_net, blobs):
    norms = {}
    for name in blobs:
        norm = np.linalg.norm(caffe_net.blobs[name].data)
        norms[name] = norm / np.sqrt(caffe_net.blobs[name].data.size)

    return norms


def extract_net_data(caffe_net, blobs):
    data = {}
    for name in blobs:
        data[name] = np.squeeze(caffe_net.blobs[name].data)

    return data

def pretty_print(param_dict):
    for param in sorted(param_dict.keys()):
        print param.ljust(19), param_dict[param]


class NetLogger:
    def __init__(self, net, path, reset=False):
        self.path = path + str(datetime.now())
        if not os.path.isdir(path):
            os.makedirs(path)

        self.net = net

    def write(self):
        # Extract relevant data from net
        data = extract_net_data(self.net, ('loss',))
        grad_norms = compute_gradient_norms(self.net)
        param_norms = compute_param_norms(self.net)

        # Append to files
        with open(os.path.join(self.path, "loss"), 'a') as fp:
            print >> fp, data['loss']

        for name in grad_norms:
            with open(os.path.join(self.path, name+'.gradnorm'), 'a') as fp:
                print >> fp, ",".join(str(x) for x in grad_norms[name])

        for name in param_norms:
            with open(os.path.join(self.path, name+'.norm'), 'a') as fp:
                print >> fp, ",".join(str(x) for x in param_norms[name])


def test():
    from barista.baristanet import BaristaNet
    baristanet = BaristaNet('models/deepq/train_val.prototxt',
                            'models/deepq/deepq16.caffemodel',
                            'Augustus')

    print "\nData Norms (before loading):"
    print "-"*40
    data_norms = compute_data_norms(baristanet.net, ('state', 'next_state'))
    pretty_print(data_norms)
    baristanet.dummy_load_minibatch()
    print "\nData Norms (after loading):"
    print "-"*40
    data_norms = compute_data_norms(baristanet.net, ('state', 'next_state'))
    pretty_print(data_norms)

    print "\nGradient norms (before computing):"
    print "-"*40
    grad_norms = compute_gradient_norms(baristanet.net, ord=2)
    pretty_print(grad_norms)

    baristanet.full_pass()

    print "\nGradient norms (after computing):"
    print "-"*40
    grad_norms = compute_gradient_norms(baristanet.net, ord=2)
    pretty_print(grad_norms)

    data = extract_net_data(
               baristanet.net,
               ('Q_sa', 'action', 'reward', 'P_sa', 'loss'))
    Q, P = data['Q_sa'], data['P_sa']
    action, reward, loss = data['action'], data['reward'], data['loss']
    print "\nQ_sa:"
    print "-"*40
    print Q.reshape((Q.size, 1))
    print "P_sa:"
    print P
    print "Action:"
    print action
    print "Reward:"
    print reward
    print "Loss:", loss

if __name__ == "__main__":
    test()
