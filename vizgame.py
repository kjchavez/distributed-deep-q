import tables as tb
import matplotlib.pyplot as plt
from scipy import misc
import matplotlib.cm as cm
from matplotlib.widgets import Slider


class Experience(tb.IsDescription):
    action = tb.IntCol(pos=1)
    reward = tb.IntCol(pos=2)


def record(action, reward, frame, newfile=False):
    if not newfile:
        with tb.open_file("history.h5", "a") as f:
            exp = f.root.exp.history
            exp.append([(action, reward)])

            frames = f.root.frames
            frame_number = frames._v_nchildren
            f.create_array(frames, "frame" + str(frame_number), frame,
                           title="frame " + str(frame_number))
    else:
        with tb.open_file("history.h5", "a") as f:
            exp = f.create_group("/", "exp", "Experience history")
            h = f.create_table(exp, "history", Experience, "(action,rewards)")
            h.append([(action, reward)])

            frames = f.create_group("/", "frames", "Frame history")
            f.create_array(frames, "frame 0", frame, title="frame0")


def get_frame(frame_number):
    with tb.open_file("history.h5", "a") as f:
        frames = f.root.frames._f_list_nodes()
        return frames[frame_number].read()


def gen_image(frame, frame_number):
    misc.imsave("frame" + str(frame_number) + ".png", frame)


def view_images(frames):
    ''' |frames| is a list of numpy arrays. '''

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l, = plt.imshow(frames[0])

    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes(list(range(len(frames))), axisbg=axcolor)
    sframe = Slider(axframe, "Frame no.", 0.1, 30.0, valinit=0)

    def update(val):
        l.set_data(frames[sframe.val])
        plt.draw()

    sframe.on_changed(update)
    plt.show()
