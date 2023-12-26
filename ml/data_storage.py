from typing import Any

import pandas as pd
from pandas import DataFrame


class DataStorage:

    def __init__(self, path: str, x: str, y: str) -> None:
        self._df = pd.read_csv(path, index_col=False)
        self._x = x
        self._y = y

    @property
    def x(self) -> DataFrame:
        return self._df[self._x]

    @property
    def y(self) -> DataFrame:
        return self._df[self._y]

    def add(self, data: list[Any]) -> None:
        s = pd.DataFrame([data], columns=self._df.columns)
        self._df = pd.concat([self._df, s], ignore_index=True)

    def preprocess(self) -> DataFrame:
        return self._df[[self._x, self._y]]

    def postprocess(self):
        pass
