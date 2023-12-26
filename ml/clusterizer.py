from typing import Sequence

from pandas.core.interchange.dataframe_protocol import DataFrame


class Clusterizer:

    def __init__(self, model, n_clusters: int) -> None:
        self._model_type = model
        self._n_clusters = n_clusters
        self._model = self._model_type(n_clusters=n_clusters, n_init="auto")

    def change_num_of_clusters(self, new_n_clusters: int) -> None:
        self._n_clusters = new_n_clusters
        self._model = self._model_type(n_clusters=new_n_clusters, n_init="auto")

    def fit(self, x: DataFrame) -> None:
        self._model.fit(x)

    def predict(self, x: DataFrame) -> Sequence:
        return self._model.predict(x)
