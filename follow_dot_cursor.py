from datetime import datetime
from datetime import date

import matplotlib.pyplot as plt
import scipy.spatial as spatial
import numpy as np

import csv
import sys
import pandas as pd


def fmt(x, y):
    x = round(float(x), 2)
    y = round(float(y), 2)
    default_date = '2012-12-23'
    df = pd.read_csv('data/processed_weather_data.csv', index_col=False)
    result = df[df['Temperature (C)'] == x]
    temperature_foundation = df[df['Temperature (C)'].astype(str).str.contains(str(x))]
    humidity_foundation = df[df['Humidity'].astype(str).str.contains(str(y))]
    if temperature_foundation.shape[0]:
        default_date = temperature_foundation.iloc[0]['Formatted Date']
    elif humidity_foundation.shape[0]:
        default_date = humidity_foundation.iloc[0]['Formatted Date']
    return f'{default_date}'.format(x=x, y=y)


class FollowDotCursor(object):

    def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        x = np.asarray(x, dtype='float')
        y = np.asarray(y, dtype='float')
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        self.offsets = offsets
        y = y[np.abs(y - y.mean()) <= 3 * y.std()]
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
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
            '', xy=(1, 1), ha='right',
            xytext=self.offsets, textcoords='offset points', va='bottom',
            bbox=dict(
                boxstyle='round,pad=0.4', fc='white', alpha=0.85),
            arrowprops=dict(
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
