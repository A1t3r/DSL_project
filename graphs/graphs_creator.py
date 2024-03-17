import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class ClusterGraphsCreator:

    @property
    def fig(self) -> Figure:
        return self._fig

    def __init__(self) -> None:
        self._fig: Figure = plt.figure()
        self._ax = self._fig.add_subplot(1, 1, 1)

    def plot(self, X, Y, labels) -> None:
        self._ax.scatter(X, Y, c=labels)
