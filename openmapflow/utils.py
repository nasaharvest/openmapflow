from pathlib import Path
from typing import List

import pandas as pd


def try_txt_read(file_path: Path) -> List[str]:
    try:
        return pd.read_csv(file_path, sep="\n", header=None)[0].tolist()
    except FileNotFoundError:
        return []


def find_project_root(files_to_check: List[str]) -> Path:
    """
    Find the project root directory by checking for existence of certain files
    """
    possible_roots = [Path.cwd(), Path.cwd().parent]

    for root in possible_roots:
        if all([(root / c).exists() for c in files_to_check]):
            return root

    raise FileExistsError(f"{files_to_check} not found in {possible_roots}.")
