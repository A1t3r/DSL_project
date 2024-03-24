import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from graphs.follow_dot_cursor import FollowDotCursor


class ClusterGraphsCreator:

    @property
    def fig(self) -> Figure:
        return self._fig

    def __init__(self) -> None:
        self._fig: Figure = plt.figure()
        self._ax = self._fig.add_subplot(1, 1, 1)

    def plot(self, X, Y, df, labels) -> None:
        FollowDotCursor(self._ax, X, Y, df, tolerance=20)
        self._ax.scatter(X, Y, c=labels)
