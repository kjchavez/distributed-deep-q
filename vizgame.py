import tables as tb
import matplotlib.pyplot as plt
from scipy import misc


class Experience(tb.IsDescription):
    action = tb.IntCol(pos=1)
    reward = tb.IntCol(pos=2)


def record(self, action, reward, frame, frame_number, newfile=False):
    if not newfile:
        with tb.open_file("history.h5", "a") as f:
            exp = f.root.exp.history
            exp.append((action, reward))

            frames = f.root.frames
            f.create_array(frames, "frame" + str(frame_number), frame,
                           title="frame " + str(frame_number))
    else:
        with tb.open_file("history.h5", "w") as f:
            exp = f.create_group("/", "exp", "Experience history")
            h = f.create_table(exp, "history", Experience, "(action,rewards)")
            h.append((action, reward))

            frames = f.create_group("/", "frames", "Frame history")
            f.create_array(frames, "frame" + str(frame_number), frame,
                           title="frame " + str(frame_number))


def get_frame(frame_number):
    with tb.open_file("history.h5", "a") as f:
        frames = f.root.frames._f_list_nodes()
        return frames[frame_number]


def gen_image(frame, frame_number):
    misc.imsave("frame" + str(frame_number) + ".png", frame)


def view_image(frame):
    plt.imshow(frame)
    plt.show()
