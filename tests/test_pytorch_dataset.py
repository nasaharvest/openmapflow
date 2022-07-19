import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from openmapflow.constants import CLASS_PROB, END, LAT, LON, START, EO_DATA

try:
    import torch  # noqa: F401

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

if TORCH_INSTALLED:
    from openmapflow.pytorch_dataset import PyTorchDataset, _upsample_df

tempdir = tempfile.gettempdir()


class TestPyTorchDataset(TestCase):

    tif_values: np.ndarray
    df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        if not TORCH_INSTALLED:
            return

        cls.tif_values = np.zeros((24, 18))

        cls.df = pd.DataFrame(
            {
                CLASS_PROB: [0.0, 1.0],
                START: ["2019-01-01", "2019-01-01"],
                END: ["2021-12-31", "2021-12-31"],
                LAT: [0.0, 1.0],
                LON: [0.0, 1.0],
                EO_DATA: [cls.tif_values, cls.tif_values],
            }
        )

    def setUp(self) -> None:
        if not TORCH_INSTALLED:
            self.skipTest("Torchserve is not installed")

    def test_df_empty_arg(self):
        df = pd.DataFrame({})
        with self.assertRaises(ValueError):
            PyTorchDataset(df)

    def test_df_missing_cols_arg(self):
        df = pd.DataFrame({CLASS_PROB: [0.5]})
        with self.assertRaises(ValueError):
            PyTorchDataset(df)

    def test_invalid_subset_arg(self):
        with self.assertRaises(ValueError):
            PyTorchDataset(self.df, subset="invalid")

    def test_invalid_start_month_arg(self):
        with self.assertRaises(ValueError):
            PyTorchDataset(self.df, start_month="Marchember")

    def test_invalid_input_months_arg(self):
        with self.assertRaises(ValueError):
            PyTorchDataset(self.df, input_months=0)

    def test_invalid_upsample_ratio(self):
        with self.assertRaises(ValueError):
            PyTorchDataset(self.df, upsample_minority_ratio=0.0)

    def test_invalid_probability_threshold(self):
        with self.assertRaises(ValueError):
            PyTorchDataset(self.df, probability_threshold=-0.1)

    def test_len(self):
        self.assertEqual(len(PyTorchDataset(self.df)), len(self.df))
        df_times_2 = pd.concat([self.df, self.df])
        self.assertEqual(len(PyTorchDataset(df_times_2)), len(self.df) * 2)
        df_times_100 = pd.concat([self.df] * 100)
        self.assertEqual(len(PyTorchDataset(df_times_100)), len(self.df) * 100)

    def test_to_array(self):
        dataset = PyTorchDataset(self.df)
        x, y, is_local = dataset.to_array()
        self.assertEqual(
            x.shape, (len(self.df), dataset.input_months, self.tif_values.shape[1])
        )
        self.assertEqual(y.shape, (len(self.df),))
        self.assertEqual(is_local.shape, (len(self.df),))

    def test_getitem(self):
        dataset = PyTorchDataset(self.df)
        x, y, is_local = dataset[0]
        self.assertEqual(x.shape, (dataset.input_months, self.tif_values.shape[1]))
        self.assertEqual(y.shape, ())
        self.assertEqual(is_local.shape, ())
        self.assertEqual(y.item(), 0.0)
        self.assertEqual(is_local.item(), 1.0)
        self.assertTrue(
            (x.detach().numpy() == self.tif_values[: dataset.input_months]).all()
        )

    def test_identity_upsample_df(self):
        original_df = PyTorchDataset(self.df).df
        upsampled_df = _upsample_df(original_df, upsample_ratio=1.0)
        self.assertEqual(len(upsampled_df), len(original_df))

    def test_simple_upsample_df(self):
        df = self.df.copy()
        df = df.append(df.iloc[-1])
        original_df = PyTorchDataset(df).df
        upsampled_df = _upsample_df(original_df, upsample_ratio=1.0)
        self.assertEqual(len(upsampled_df), len(original_df) + 1)

    def test_ratio_upsample_df(self):
        df = self.df.copy()
        df = df.append([df[df[CLASS_PROB] == 1.0]] * 9)
        df = PyTorchDataset(df).df
        upsampled_df = _upsample_df(df, upsample_ratio=0.2)
        self.assertEqual(len(upsampled_df) - len(df), 1)
        upsampled_df = _upsample_df(df, upsample_ratio=0.5)
        self.assertEqual(len(upsampled_df) - len(df), 4)
        upsampled_df = _upsample_df(df, upsample_ratio=0.8)
        self.assertEqual(len(upsampled_df) - len(df), 7)
        upsampled_df = _upsample_df(df, upsample_ratio=0.666)
        self.assertEqual(len(upsampled_df) - len(df), 6)

    def test_ratio_upsample_dataset(self):
        df = self.df.copy()
        df = df.append([df[df[CLASS_PROB] == 1.0]] * 9)
        upsampled_df = PyTorchDataset(df, upsample_minority_ratio=0.2)
        self.assertEqual(len(upsampled_df) - len(df), 1)
        upsampled_df = PyTorchDataset(df, upsample_minority_ratio=0.5)
        self.assertEqual(len(upsampled_df) - len(df), 4)
        upsampled_df = PyTorchDataset(df, upsample_minority_ratio=0.8)
        self.assertEqual(len(upsampled_df) - len(df), 7)
        upsampled_df = PyTorchDataset(df, upsample_minority_ratio=0.666)
        self.assertEqual(len(upsampled_df) - len(df), 6)
