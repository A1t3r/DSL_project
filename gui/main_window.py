import tkinter as tk
import tksheet
from typing import Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from graphs.graphs_creator import ClusterGraphsCreator
from ml.clusterizer import Clusterizer
from ml.data_storage import DataStorage
from utils.datamodels import WindowSettings, ClusterizerSettings, DataSettings


class MainWindow:

    def __init__(self, *args,
                 ws: WindowSettings,
                 cs: ClusterizerSettings,
                 ds: DataSettings
                 ) -> None:
        self._root = tk.Tk()
        self._root.title(ws.title)
        self._root.geometry(ws.geometry)

        self._data_storage = DataStorage(path=ds.path, x=ds.x_name, y=ds.y_name)
        self._clusterizer = Clusterizer(model=cs.model, n_clusters=cs.n_clusters)
        self._cluster_graphs_creator = ClusterGraphsCreator()

        # Change number of clusters
        tk.Label(text='Number of clusters').pack()
        self._n_clusters_entry= tk.Entry()
        self._n_clusters_entry.pack(padx=6, pady=6)
        tk.Button(text="Change number of clusters", command=self._change_n_clusters).pack()

        #  Display graph
        tk.Button(text="Display Graph", command=self._display_graph).pack()

        #  Draw Graph
        self._canvas = FigureCanvasTkAgg(self._cluster_graphs_creator.fig)
        self._canvas.get_tk_widget().pack(side=tk.RIGHT)

        #  Add data dynamically
        self._data_entry_x = tk.Entry()
        self._data_entry_x.pack(side=tk.BOTTOM, padx=6, pady=6)
        tk.Label(text=ds.x_name).pack(side=tk.BOTTOM)

        self._data_entry_y = tk.Entry()
        self._data_entry_y.pack(side=tk.BOTTOM, padx=6, pady=6)
        tk.Label(text=ds.y_name).pack(side=tk.BOTTOM)

        tk.Button(text="Add data", command=self._add_data).pack(side=tk.BOTTOM)

        #  Dump data
        tk.Button(text="Dump data", command=self._dump_data).pack(side=tk.BOTTOM)

        # Load data
        tk.Button(text="Load data", command=self._load_data).pack(side=tk.BOTTOM)

        #  Display DataFrame
        self._sheet = tksheet.Sheet(self._root)
        self._sheet.pack(side=tk.LEFT)
        self._sheet.headers([ds.x_name, ds.y_name])
        self._sheet.set_sheet_data(list(self._data_storage.preprocess().to_numpy()))

    def _run_predict_cycle(self) -> Sequence:
        X = self._data_storage.preprocess()
        self._clusterizer.fit(X)
        labels = self._clusterizer.predict(X)
        return labels

    def _display_graph(self) -> None:
        labels = self._run_predict_cycle()
        self._cluster_graphs_creator.plot(
            self._data_storage.x, self._data_storage.y, labels=labels
        )
        self._canvas.draw()

    def _add_data(self) -> None:
        x_val = self._data_entry_x.get()
        y_val = self._data_entry_y.get()
        #  TODO: Data display and find decent solution for id
        self._data_storage.add([None, None, x_val, y_val])
        self._sheet.set_sheet_data(list(self._data_storage.preprocess().to_numpy()))

    def _change_n_clusters(self) -> None:
        n_clusters = self._n_clusters_entry.get()
        self._clusterizer.change_num_of_clusters(int(n_clusters))

    def _dump_data(self) -> None:
        #  TODO
        pass

    def _load_data(self) -> None:
        #  TODO
        pass

    def mainloop(self) -> None:
        self._root.mainloop()
