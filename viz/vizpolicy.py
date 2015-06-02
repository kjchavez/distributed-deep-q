import argparse
import cPickle
import os
from barista import baristanet
from barista import netutils
from replay import ReplayDataset
from expgain import *
from gamesim.SnakeGame import SnakeGame, gray_scale
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("model")
    parser.add_argument("checkpoint")
    parser.add_argument("--num-games", '-n', type=int, default=5)
    parser.add_argument("--epsilon", '-e', type=float, default=0.05)
    parser.add_argument("--output-dir", '-o', default="frames")

    args = parser.parse_args()
    return args


def load_saved_checkpoint(filename):
    with open(filename) as fp:
        params = cPickle.load(fp)

    return params


def main():
    args = get_args()

    # Instantiate network
    bnet = baristanet.BaristaNet(args.architecture, args.model, None)

    # load parameters from checkpoint into the model
    # params = load_saved_checkpoint(args.checkpoint)
    # netutils.set_net_params(bnet.net, params)

    # Initialize game player
    replay_dataset = ReplayDataset("temp-dset.hdf5", bnet.state[0].shape,
                                   dset_size=300*args.num_games,
                                   overwrite=True)

    game = SnakeGame()
    preprocessor = generate_preprocessor(bnet.state.shape[2:], gray_scale)
    exp_gain = ExpGain(bnet, ['w', 'a', 's', 'd'], preprocessor, game.cpu_play,
                       replay_dataset, game.encode_state())

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate experiences
    frame_index = 0
    num_games_played = 0
    state = exp_gain.get_preprocessed_state()
    while num_games_played < args.num_games:
        # select action
        if random.random() < args.epsilon:
            action = random.choice(exp_gain.actions)
        else:
            idx = bnet.select_action([state])
            action = exp_gain.actions[idx]

        exp_gain.play_action(action)

        # Render frame
        frame = gray_scale(exp_gain.sequence[-1].reshape((1,)+exp_gain.sequence[-1].shape))[-1]

        big_frame = cv2.resize(frame, (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST) 
        cv2.imwrite(os.path.join(args.output_dir, "frame-%d.png" % frame_index), big_frame)
        frame_index += 1
        # cv2.imshow("Game", frame)
        # cv2.waitKey(33)

        # Check if Snake has died
        if exp_gain.game_over:
            print "Game Over"
            exp_gain.reset_game()
            num_games_played += 1

        # Get next state
        state = exp_gain.get_preprocessed_state()

if __name__ == "__main__":
    main()
