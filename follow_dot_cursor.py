from datetime import datetime
from datetime import date

import matplotlib.pyplot as plt
import scipy.spatial as spatial
import numpy as np

import csv
import sys


def fmt(x, y):
    csv_file = csv.reader(open('data/processed_weather_data.csv', "r"),
                          delimiter=',')
    x = round(float(x), 6)
    y = round(float(y), 6)
    for row in csv_file:
        flag = False
        for row in csv_file:
            if not flag:
                flag = True
                continue
            else:
                if x == round(float(row[2]), 6) and y == round(float(row[3]), 6):
                    return f'{date.fromisoformat(row[1].split(" ")[0])}'.format(x=x, y=y)

class FollowDotCursor(object):

    def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        x = np.asarray(x, dtype='float')
        y = np.asarray(y, dtype='float')
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        self.offsets = offsets
        self.x_used = 1
        self.y_used = 1
        y = y[np.abs(y-y.mean()) <= 3*y.std()]
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='0.4', alpha=0.9)
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(fmt(x, y))
        self.dot.set_offsets(np.column_stack((x, y)))
        bbox = self.annotation.get_window_extent()
        self.fig.canvas.blit(bbox)
        self.fig.canvas.draw_idle()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(1,1), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.4', fc='white', alpha=0.85),
            arrowprops = dict(
                arrowstyle='-|>', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]

        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]
