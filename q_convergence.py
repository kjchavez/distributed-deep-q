import numpy as np

from barista.baristanet import BaristaNet
from barista.netutils import set_net_params
from expgain import ExpGain, generate_preprocessor
from gamesim.SnakeGame import SnakeGame, gray_scale

from replay import ReplayDataset

from redis import Redis
import redis_collections as redisC
import argparse


def evaluate_model(barista_net, model, num_batches):
    set_net_params(barista_net.net, model)
    avg_q = 0
    for _ in xrange(num_batches):
        barista_net.load_minibatch()
        barista_net.forward(end='Q_out')
        avg_q += np.mean(barista_net.blobs['Q_out'].data)
        # print barista_net.blobs['Q_out'].data.squeeze()

    avg_q /= num_batches
    return avg_q


def run(architecture_file, model_file, num_batches=10, pattern="centralModel-*"):
    print "evaluating Q values..."
    redisInstance = Redis(host='localhost', port=6379, db=0)
    model_keys = redisInstance.keys(pattern)
    results = {}

    net = BaristaNet(architecture_file, model_file, None)
    replay_dataset = ReplayDataset("temp-q-converge-dset.hdf5",
                                   net.state[0].shape,
                                   dset_size=1000,
                                   overwrite=True)
    net.add_dataset(replay_dataset)

    game = SnakeGame()
    preprocessor = generate_preprocessor(net.state.shape[2:], gray_scale)
    exp_gain = ExpGain(net, ['w', 'a', 's', 'd'], preprocessor, game.cpu_play,
                       replay_dataset, game.encode_state())

    print "Generating new experiences..."
    for _ in xrange(100):
        exp_gain.generate_experience(1e5)
    print "Done"

    for key in model_keys:
        print "Evaluating model:", key
        model = dict(redisC.Dict(key=key, redis=redisInstance))
        q_avg = evaluate_model(net, model, num_batches)
        results[key] = q_avg

    for key in sorted(results.keys()):
        print key.ljust(25) + "%0.4f" % results[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("model")
    parser.add_argument("--num-batches", '-n', type=int, default=10)

    args = parser.parse_args()
    run(args.architecture, args.model, num_batches=args.num_batches)

if __name__ == "__main__":
    main()
