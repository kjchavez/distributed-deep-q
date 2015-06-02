import os
from replay import ReplayDataset
import argparse
import cv2

def view_dset(rdset, frame_rate=25):
    for i in xrange(rdset.dset_size):
        cv2.imshow("Game", rdset.state[i][-1])
        print rdset.non_terminal[i]
        if not rdset.non_terminal[i]:
            print "ENDGAME at idx", i
        cv2.waitKey(1000/frame_rate)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dset")
    parser.add_argument("--frame-rate", '-r', type=int, default=25)

    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.dset):
        print "Dataset does not exist."
        return

    rdset = ReplayDataset(args.dset, None, overwrite=False)
    view_dset(rdset, frame_rate=args.frame_rate)

if __name__ == "__main__":
    main()
