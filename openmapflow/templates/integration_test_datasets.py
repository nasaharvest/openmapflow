import os
import unittest
from datetime import date
from unittest import TestCase

import pandas as pd
from datasets import datasets
from dateutil.relativedelta import relativedelta

from openmapflow.constants import (
    EO_DATA,
    EO_FILE,
    EO_LAT,
    EO_LON,
    LAT,
    LON,
    START,
    SUBSET,
)
from openmapflow.labeled_dataset import get_label_timesteps, verify_df


class IntegrationTestLabeledData(TestCase):
    """Tests that the datasets look right"""

    dfs: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        dfs = []
        if len(datasets) == 0:
            print("WARNING: No datasets found. Skipping all tests.")
            raise unittest.SkipTest("No datasets found. Skipping all tests.")

        for d in datasets:
            df = d.load_df(to_np=True)
            df["name"] = d.name
            dfs.append(df)
        cls.dfs = pd.concat(dfs)

    def test_raw_label_conversion(self):
        for d in datasets:
            print(d.name)
            verify_df(d.load_labels())

    def test_dataset_subset_amounts(self):
        all_subsets_correct_size = True

        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.name]
            label_subset_counts = df[SUBSET].value_counts()
            eo_data_subset_counts = df[df[EO_DATA].notnull()][SUBSET].value_counts()
            for subset in df[SUBSET].unique():
                label_subset_count = label_subset_counts.get(subset, 0)
                eo_data_subset_count = eo_data_subset_counts.get(subset, 0)
                if label_subset_count != eo_data_subset_count:
                    all_subsets_correct_size = False

        self.assertTrue(
            all_subsets_correct_size,
            "Check report.txt for which subsets have different sizes.",
        )

    def test_for_duplicates(self):
        duplicates = self.dfs[self.dfs.duplicated(subset=[EO_LAT, EO_LON, EO_FILE])]
        num_dupes = len(duplicates)
        self.assertTrue(num_dupes == 0, f"Found {num_dupes} duplicates")

    def test_eo_data_for_emptiness(self):
        num_empty_eo_data = len(self.dfs[self.dfs[EO_DATA].isnull()])
        self.assertTrue(
            num_empty_eo_data == 0,
            f"Found {num_empty_eo_data} empty eo_data, run openmapflow create-datasets",
        )

    def test_all_eo_data_has_18_bands(self):
        is_empty = self.dfs[EO_DATA].isnull()
        band_amount = self.dfs[~is_empty][EO_DATA].apply(lambda f: f.shape[-1]).unique()
        self.assertEqual(band_amount.tolist(), [18], f"Found {band_amount} bands")

    def test_label_and_eo_data_ranges_match(self):
        all_label_and_eo_data_ranges_match = True
        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.name]
            label_month_amount = get_label_timesteps(df)
            eo_data_month_amount = df[EO_DATA].apply(lambda f: f.shape[0])

            if (eo_data_month_amount == label_month_amount).all():
                mark = "\u2714"
                last_word = "match"
            else:
                mark = "\u2716"
                last_word = "mismatch"
                all_label_and_eo_data_ranges_match = False

            label_ranges = label_month_amount.value_counts().to_dict()
            eo_data_ranges = eo_data_month_amount.value_counts().to_dict()
            print(
                f"{mark} {d.name} label {label_ranges} and "
                + f"eo_data {eo_data_ranges} ranges {last_word}"
            )
        self.assertTrue(
            all_label_and_eo_data_ranges_match,
            "Check logs for which subsets have different sizes.",
        )

    def test_all_older_eo_data_has_24_months(self):

        current_cutoff_date = date.today().replace(day=1) + relativedelta(months=-3)
        two_years_before_cutoff = pd.Timestamp(
            current_cutoff_date + relativedelta(months=-24)
        )

        all_older_eo_data_has_24_months = True

        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.name]
            cutoff = pd.to_datetime(df[START]) < two_years_before_cutoff
            df = df[cutoff].copy()
            if len(df) == 0:
                continue
            month_amount = (
                df[df[EO_DATA].notnull()][EO_DATA].apply(lambda f: f.shape[0]).unique()
            )

            if month_amount.tolist() == [24]:
                mark = "\u2714"
            else:
                all_older_eo_data_has_24_months = False
                mark = "\u2716"
            print(f"{mark} {d.name} \t\t{month_amount.tolist()}")

        self.assertTrue(
            all_older_eo_data_has_24_months,
            "Not all older earth observation data has 24 months, check logs.",
        )

    def test_label_eo_data_for_closeness(self):
        total_num_mismatched = 0
        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.name]

            if len(df) == 0:
                print(f"\\ {d.name}:\t\tNo data")
                continue

            label_tif_mismatch = df[
                ((df[LON] - df[EO_LON]) > 0.0001) | ((df[LAT] - df[EO_LAT]) > 0.0001)
            ]
            num_mismatched = len(label_tif_mismatch)
            if num_mismatched > 0:
                mark = "\u2716"
            else:
                mark = "\u2714"
            print(f"{mark} {d.name}:\t\tMismatches: {num_mismatched}")
            total_num_mismatched += num_mismatched
        self.assertTrue(
            total_num_mismatched == 0,
            f"Found {total_num_mismatched} mismatched labels+tifs.",
        )

    def test_label_coordinate_duplication(self):
        """For now this test is just a status report"""

        duplicates = self.dfs[self.dfs.duplicated(subset=[LON, LAT], keep=False)]
        duplicates["start_year"] = pd.to_datetime(duplicates[START]).dt.year.astype(str)
        df = duplicates.groupby([LON, LAT], as_index=False, sort=False).agg(
            {
                "name": lambda names: ",".join(names.unique()),
                SUBSET: lambda subs: ",".join(subs.unique()),
                "start_year": lambda start_years: ",".join(start_years),
            }
        )
        print("------------------------------------------------------")
        print("Label coordinate spill over")
        print("------------------------------------------------------")
        print(df[["name", SUBSET, "start_year"]].value_counts())


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
