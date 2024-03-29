import tkinter as tk
from tkinter import ttk, messagebox
from typing import Sequence
import tksheet
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from graphs.graphs_creator import ClusterGraphsCreator
from ml.clusterizer import Clusterizer
from ml.data_storage import DataStorage
from utils.datamodels import WindowSettings, ClusterizerSettings, DataSettings
from PIL import Image, ImageTk


class MainWindow:

    def __init__(self,
                 ws: WindowSettings,
                 cs: ClusterizerSettings,
                 ds: DataSettings) -> None:

        self.root = None
        self._root = tk.Tk()
        self._root.title(ws.title)
        self._root.geometry(ws.geometry)
        self._root.configure(bg="white")

        self._data_storage = DataStorage(path=ds.path, x=ds.x_name, y=ds.y_name)
        self._clusterizer = Clusterizer(model=cs.model, n_clusters=cs.n_clusters)
        self._cluster_graphs_creator = ClusterGraphsCreator()

        #  Define protocol handlers
        #  Close window and destroy Python process
        self._root.protocol("WM_DELETE_WINDOW", self._on_closing)

        image1 = Image.open("img/kalendar.png")
        image1 = image1.resize((60, 60))
        self._image1 = ImageTk.PhotoImage(image1)
        label1 = tk.Label(self._root, image=self._image1)
        label1.image = self._image1
        label1.place(x=665, y=30)
        label1.configure(bg="white")

        image2 = Image.open("img/thermometer.png")
        image2 = image2.resize((30, 90))
        self._image2 = ImageTk.PhotoImage(image2)
        label2 = tk.Label(self._root, image=self._image2)
        label2.image = self._image2
        label2.place(x=680, y=115)
        label2.configure(bg="white")

        image3 = Image.open("img/water.png")
        image3 = image3.resize((60, 60))
        self._image3 = ImageTk.PhotoImage(image3)
        label3 = tk.Label(self._root, image=self._image3)
        label3.image = self._image3
        label3.place(x=665, y=250)
        label3.configure(bg="white")

        image4 = Image.open("img/sunny.png")
        image4 = image4.resize((60, 60))
        self._image4 = ImageTk.PhotoImage(image4)
        label4 = tk.Label(self._root, image=self._image4)
        label4.image = self._image4
        label4.place(x=1035, y=690)
        label4.configure(bg="white")

        image5 = Image.open("img/sunny1.png")
        image5 = image5.resize((60, 60))
        self._image5 = ImageTk.PhotoImage(image5)
        label5 = tk.Label(self._root, image=self._image5)
        label5.image = self._image5
        label5.place(x=1555, y=690)
        label5.configure(bg="white")

        #  Change number of clusters
        label = tk.Label(self._root, text="Необходимое число кластеров :", font=('Arial', 30), bg="white")
        label.place(x=1100, y=700)

        self._n_clusters_entry = tk.Entry(font=('Arial', 26))
        self._n_clusters_entry.place(x=1150, y=800)

        label = tk.Label(self._root, text="Дата наблюдения:", font=('Arial', 30), bg="white")
        label.place(x=30, y=40)

        b5 = tk.Button(self._root, text="Изменить количество кластеров", command=self._change_n_clusters, font=('Arial', 30))
        b5.place(x=1060, y=880)

        #  Display graph
        b4 = tk.Button(self._root, text="Отобразить график", command=self._display_graph, font=('Arial', 30))
        b4.place(x=1150, y=960)

        #  Draw Graph
        self._canvas = FigureCanvasTkAgg(self._cluster_graphs_creator.fig)
        self._canvas.get_tk_widget().place(x=910, y=10)

        #  Add data dynamically
        self._data_entry_x = ttk.Entry(font=('Arial', 26))
        self._data_entry_x.place(x=350, y=148)
        label = tk.Label(self._root, text="Ввод температуры:", font=('Arial', 30), bg="white")
        label.place(x=30, y=150)

        self._data_entry_y = tk.Entry(font=('Arial', 26))
        self._data_entry_y.place(x=350, y=258)

        label = tk.Label(self._root, text="Ввод влажности:", font=('Arial', 30), bg="white")
        label.place(x=30, y=260)

        b1 = tk.Button(self._root, text="Добавить данные", command=self._add_data, font=('Arial', 30))
        b1.place(x=170, y=350, height=50, width=280)

        #  Dump data
        b2 = tk.Button(self._root, text="Открыть файл", command=self._dump_data, font=('Arial', 30))
        b2.place(x=170, y=710, height=50, width=280)

        b3 = tk.Button(text="Сохранить в файл", command=self._load_data, font=('Arial', 30))
        b3.place(x=170, y=900, height=50, width=280)

        #  Display DataFrame
        self._sheet = tksheet.Sheet(self._root)
        self._sheet.place(x=10, y=440)
        self._sheet.headers([ds.x_name, ds.y_name])
        self._sheet.set_sheet_data(list(self._data_storage.preprocess().to_numpy()))


    def _run_predict_cycle(self) -> Sequence:
        x = self._data_storage.preprocess()
        self._clusterizer.fit(x)
        labels = self._clusterizer.predict(x)
        return labels

    def _display_graph(self) -> None:
        labels = self._run_predict_cycle()
        self._cluster_graphs_creator.plot(
            self._data_storage.x, self._data_storage.y, self._data_storage.df, labels=labels
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

    def _on_closing(self) -> None:
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self._root.quit()

    def mainloop(self) -> None:
        self._root.mainloop()
