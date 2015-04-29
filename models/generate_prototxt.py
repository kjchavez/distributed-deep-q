# Create .prototxt from template
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("output")
parser.add_argument("num_actions", type=int)
parser.add_argument("--gamma", type=float, default=0.85)
parser.add_argument("--template", default="deepq/train_val.template")

args = parser.parse_args()

with open(args.template) as template:
    with open(args.output, 'w') as prototxt:
        for line in template:
            line = line.rstrip()
            if "<GAMMA>" in line:
                newline = line.replace("<GAMMA>", str(args.gamma))
                print >> prototxt, newline
            elif "<NUM_ACTIONS>" in line:
                newline = line.replace("<NUM_ACTIONS>", str(args.num_actions))
                print >> prototxt, newline
            elif "<ACTION_REPEAT> " in line:
                newline = line.replace("<ACTION_REPEAT> ", '')
                num_repeat = args.num_actions
                if "slice_point" in newline:
                    num_repeat = args.num_actions - 1
                for i in xrange(num_repeat):
                    print >> prototxt, newline % (i+1)
            else:
                print >> prototxt, line
