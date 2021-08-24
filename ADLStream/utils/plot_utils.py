"""Utilities for result and progress visualization."""

import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, clear_output
import time


class EvaluationVisualizer:
    """Evaluation Visualizer

    This class handle all the work related with plotting the results.
    It is used by the evaluators.

    Arguments:
        title (str): Figure title.
        ylabel (str): Metric name.
    """

    def __init__(self, title, ylabel):
        self.title = title
        self.ylabel = ylabel

        self.xdata = []
        self.ydata = []

        self._last_draw = time.time()

    def start(self):
        self._initialize_plot()

    def _initialize_plot(self):
        self.fig = plt.figure()
        self.fig.suptitle(f"ADLStream - {self.title}")
        self.ax = self.fig.add_subplot(1, 1, 1)

    def _update_plot(self):
        # Set max of 4 frames per second
        if time.time() - self._last_draw < 0.25:
            return

        self._last_draw = time.time()

        self.ax.relim()
        self.ax.cla()
        self.ax.plot(self.xdata, self.ydata)
        self.ax.legend(labels=[f"{self.ylabel} ({self.ydata[-1]:.4f})"])

        if InteractiveShell.initialized():
            # Support for notebook backend.
            print("ADLStream", end="")  # Needed due to a bug (StackOverflow #66176016).
            display(self.fig)
            clear_output(wait=True)

        plt.pause(1e-9)

    def append_data(self, x, y):
        self.xdata += x
        self.ydata += y
        self._update_plot()

    def savefig(self, filename):
        self.fig.savefig(filename)

    def show(self):
        if InteractiveShell.initialized():
            clear_output()
            display(self.fig)
        else:
            plt.show()
