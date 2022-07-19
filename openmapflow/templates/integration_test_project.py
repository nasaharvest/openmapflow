import os
import unittest
from datetime import date
from unittest import TestCase

import pandas as pd

from datasets import datasets
from dateutil.relativedelta import relativedelta

from openmapflow.constants import (
    LAT,
    LON,
    START,
    SUBSET,
    EO_DATA,
    EO_FILE,
    EO_LAT,
    EO_LON,
)

from openmapflow.labeled_dataset import get_label_timesteps


class IntegrationTestLabeledData(TestCase):
    """Tests that the features look right"""

    @classmethod
    def setUpClass(cls) -> None:
        dfs = []
        for d in datasets:
            df = d.load_df()
            df["name"] = d.dataset
            dfs.append(df)
        cls.dfs = pd.concat(dfs)

    def test_label_feature_subset_amounts(self):
        all_subsets_correct_size = True

        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.dataset]
            label_subset_counts = df[SUBSET].value_counts()
            eo_data_subset_counts = df[df[EO_DATA].notnull()][SUBSET].value_counts()

            print(d.summary(df))
            for subset in df[SUBSET].unique():
                label_subset_count = label_subset_counts.get(subset, 0)
                eo_data_subset_count = eo_data_subset_counts.get(subset, 0)
                if label_subset_count != eo_data_subset_count:
                    all_subsets_correct_size = False

        self.assertTrue(
            all_subsets_correct_size,
            "Check logs for which subsets have different sizes.",
        )

    def test_for_duplicates(self):
        duplicates = self.dfs[self.dfs.duplicated(subset=[EO_LAT, EO_LON, EO_FILE])]
        num_dupes = len(duplicates)
        self.assertTrue(num_dupes == 0, f"Found {num_dupes} duplicates")

    def test_features_for_emptiness(self):
        num_empty_features = len(self.dfs[self.dfs[EO_DATA].isnull()])
        self.assertTrue(
            num_empty_features == 0,
            f"Found {num_empty_features} empty features, run create_all_features() to fix.",
        )

    def test_all_features_have_18_bands(self):
        is_empty = self.dfs[EO_DATA].isnull()
        band_amount = self.dfs[~is_empty][EO_DATA].apply(lambda f: f.shape[-1]).unique()
        self.assertEqual(band_amount.tolist(), [18], "Found {band_amount} bands")

    def test_label_and_feature_ranges_match(self):
        all_label_and_feature_ranges_match = True
        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.dataset]
            label_month_amount = get_label_timesteps(df)
            feature_month_amount = df[EO_DATA].apply(lambda f: f.shape[0])

            if (feature_month_amount == label_month_amount).all():
                mark = "\u2714"
                last_word = "match"
            else:
                mark = "\u2716"
                last_word = "mismatch"
                all_label_and_feature_ranges_match = False

            label_ranges = label_month_amount.value_counts().to_dict()
            feature_ranges = feature_month_amount.value_counts().to_dict()
            print(
                f"{mark} {d.dataset} label {label_ranges} and "
                + f"feature {feature_ranges} ranges {last_word}"
            )
        self.assertTrue(
            all_label_and_feature_ranges_match,
            "Check logs for which subsets have different sizes.",
        )

    def test_all_older_features_have_24_months(self):

        current_cutoff_date = date.today().replace(day=1) + relativedelta(months=-3)
        two_years_before_cutoff = pd.Timestamp(
            current_cutoff_date + relativedelta(months=-24)
        )

        all_older_features_have_24_months = True

        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.dataset]
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
                all_older_features_have_24_months = False
                mark = "\u2716"
            print(f"{mark} {d.dataset} \t\t{month_amount.tolist()}")

        self.assertTrue(
            all_older_features_have_24_months,
            "Not all older features have 24 months, check logs.",
        )

    def test_features_for_closeness(self):
        total_num_mismatched = 0
        for d in datasets:
            df = self.dfs[self.dfs["name"] == d.dataset]

            if len(df) == 0:
                print(f"\\ {d.dataset}:\t\tNo features")
                continue

            label_tif_mismatch = df[
                ((df[LON] - df[EO_LON]) > 0.0001) | ((df[LAT] - df[EO_LAT]) > 0.0001)
            ]
            num_mismatched = len(label_tif_mismatch)
            if num_mismatched > 0:
                mark = "\u2716"
            else:
                mark = "\u2714"
            print(f"{mark} {d.dataset}:\t\tMismatches: {num_mismatched}")
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
