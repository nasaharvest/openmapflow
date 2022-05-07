from pathlib import Path
from typing import List

import pandas as pd


def try_txt_read(file_path: Path) -> List[str]:
    try:
        return pd.read_csv(file_path, sep="\n", header=None)[0].tolist()
    except FileNotFoundError:
        return []
