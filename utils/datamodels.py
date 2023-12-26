from dataclasses import dataclass
from typing import Any


@dataclass
class WindowSettings:
    title: str
    geometry: str


@dataclass
class ClusterizerSettings:
    model: Any
    n_clusters: int


@dataclass
class DataSettings:
    path: str
    x_name: str
    y_name: str
