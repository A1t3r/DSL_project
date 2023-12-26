import yaml
from sklearn.cluster import KMeans

from gui.main_window import MainWindow
from utils.datamodels import WindowSettings, ClusterizerSettings, DataSettings

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

ws = WindowSettings(title=config['WINDOW_SETTINGS']['TITLE'],
                    geometry=config['WINDOW_SETTINGS']['SIZE'])

cs = ClusterizerSettings(model=KMeans, n_clusters=config['CLUSTERIZER_SETTINGS']['N_CLUSTERS'])

ds = DataSettings(path=config['DATA_SETTINGS']['PATH'],
                  x_name=config['DATA_SETTINGS']['X_NAME'],
                  y_name=config['DATA_SETTINGS']['Y_NAME'])

mw = MainWindow(ws=ws, cs=cs, ds=ds)
mw.mainloop()
