import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from openmapflow.constants import CLASS_PROB, COUNTRY, END, EO_DATA, LAT, LON, START
from openmapflow.train_utils import generate_model_name, get_x_y, upsample_df


class TestTrainUtils(TestCase):

    tif_values: np.ndarray
    df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.tif_values = str(np.zeros((24, 18)).tolist())

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

    def test_identity_upsample_df(self):
        upsampled_df = upsample_df(self.df, upsample_ratio=1.0)
        self.assertEqual(len(upsampled_df), len(self.df))

    def test_simple_upsample_df(self):
        df = self.df.append(self.df.iloc[-1])
        upsampled_df = upsample_df(df, upsample_ratio=1.0)
        self.assertEqual(len(upsampled_df), len(df) + 1)

    def test_ratio_upsample_df(self):
        df = self.df.copy()
        df = df.append([df[df[CLASS_PROB] == 1.0]] * 9)
        upsampled_df = upsample_df(df, upsample_ratio=0.2)
        self.assertEqual(len(upsampled_df) - len(df), 1)
        upsampled_df = upsample_df(df, upsample_ratio=0.5)
        self.assertEqual(len(upsampled_df) - len(df), 4)
        upsampled_df = upsample_df(df, upsample_ratio=0.8)
        self.assertEqual(len(upsampled_df) - len(df), 7)
        upsampled_df = upsample_df(df, upsample_ratio=0.666)
        self.assertEqual(len(upsampled_df) - len(df), 6)

    def test_get_x_y(self):
        x, y = get_x_y(self.df)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), len(self.df))
        self.assertEqual(x[0].shape, (12, 18))
        self.assertEqual(y[0], 0.0)
        self.assertEqual(y[1], 1.0)

    def test_generate_model_name(self):
        self.assertEqual(generate_model_name(self.df), "openmapflow_2019")
        self.df[COUNTRY] = "Pangea"
        self.assertEqual(generate_model_name(self.df), "Pangea_openmapflow_2019")
        self.assertEqual(
            generate_model_name(self.df, start_month="February"),
            "Pangea_openmapflow_2019_February",
        )
