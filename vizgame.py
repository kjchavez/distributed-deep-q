import tables as tb
import matplotlib.pyplot as plt
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


def get_frames():
    with tb.open_file("history.h5", "a") as f:
        frames = f.root.frames._f_list_nodes()
        return [frames[fidx].read() for fidx in range(len(frames))]


class DiscreteSlider(Slider):
    """ A matplotlib slider widget with discrete steps. """

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.5)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)


def view_images(frames):
    """ |frames| is a list of numpy arrays. """

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l = plt.imshow(frames[0], cmap=cm.Greys_r, interpolation="none")

    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    sframe = DiscreteSlider(axframe, "Frame no.", 0, len(frames) - 1,
                            increment=1, valinit=0)

    def update(val):
        l.set_data(frames[int(sframe.val)])
        plt.draw()

    sframe.on_changed(update)
    plt.show()
