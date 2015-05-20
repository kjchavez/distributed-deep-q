from barista.baristanet import BaristaNet
from barista.netutils import set_net_params
from expgain import ExpGain, generate_preprocessor
from gamesim.SnakeGame import SnakeGame, gray_scale

from redis import Redis
import redis_collections as redisC
import argparse


class PolicyEvaluator(object):
    def __init__(self, architecture_file, model_file):
        # Initialize objects
        self.net = BaristaNet(architecture_file, model_file, None)
        self.batch_size = self.net.batch_size

        game = SnakeGame()
        preprocessor = generate_preprocessor(self.net.state.shape[2:],
                                             gray_scale)
        self.engines = [ExpGain(self.net, ['w', 'a', 's', 'd'],
                                preprocessor, game.cpu_play,
                                None, game.encode_state())
                        for _ in range(self.batch_size)]

    def evaluate(self, model, num_trials):
        """ Runs |num_trials| games and returns average score. """
        for eg in self.engines:
            set_net_params(eg.net.net, model)
            eg.reset_game()

        total_score = 0
        trials_completed = 0
        scores = [0] * self.batch_size
        while trials_completed < num_trials:
            states = [eg.get_preprocessed_state() for eg in self.engines]
            print "[EVALUATE]", states[0]
            actions = self.net.select_action(states,
                                             batch_size=self.batch_size)
            for i, (action, eg) in enumerate(zip(actions, self.engines)):
                scores[i] += eg.play_action(eg.actions[action])
                if eg.game_over:
                    total_score += scores[i]
                    trials_completed += 1
                    if trials_completed == num_trials:
                        break
                    eg.reset_game()
                    scores[i] = 0

        return float(total_score)/num_trials


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_file")
    parser.add_argument("model_file")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--num-trials", '-n', dest="num_trials",
                        type=int, default=32)
    parser.add_argument("--model-pattern", '-m', dest="pattern",
                        default="centralModel*")

    return parser.parse_args()


def main():
    args = get_args()
    redisInstance = Redis(host='localhost', port=6379, db=0)
    model_keys = redisInstance.keys(args.pattern)
    results = redisC.Dict(key="averageReward", redis=redisInstance)

    pe = None
    for key in model_keys:
        if key not in results or args.recompute:
            if pe is None:
                pe = PolicyEvaluator(args.architecture_file, args.model_file)
            print "Evaluating model:", key
            model = dict(redisC.Dict(key=key, redis=redisInstance))
            avg_reward = pe.evaluate(model, args.num_trials)
            results[key] = avg_reward
            print "Average reward:", avg_reward

if __name__ == "__main__":
    main()
