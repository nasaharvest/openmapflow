from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class DataInstance:
    instance_lat: float
    instance_lon: float
    labelled_array: Union[float, np.ndarray]
    source_file: str
