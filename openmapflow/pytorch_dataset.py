from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PyTorchDataset(Dataset):
    """Used for training and evaluating PyTorch based models."""

    def __init__(self, x: List[np.ndarray], y: List[float]) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x_tensor = torch.from_numpy(self.x[index]).float()
        y_tensor = torch.tensor(int(self.y[index])).float()
        return x_tensor, y_tensor
