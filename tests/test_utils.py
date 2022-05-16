from datetime import date
from unittest import TestCase

import numpy as np
import pandas as pd

from openmapflow.utils import to_date


class TestUtils(TestCase):
    def test_to_date(self):
        np_date = np.datetime64("2020-01-01")
        str_date = "2020-01-01"
        df_date = pd.to_datetime("2020-01-01")
        obj_date = date(2020, 1, 1)
        self.assertEqual(to_date(np_date), obj_date)
        self.assertEqual(to_date(str_date), obj_date)
        self.assertEqual(to_date(df_date), obj_date)
