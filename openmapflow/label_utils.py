import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.compat._optional import import_optional_dependency


def train_val_test_split(
    index: pd.RangeIndex, val: float = 0.0, test: float = 0.0
) -> pd.DataFrame:
    """Splits a series into train, val and test"""
    if val + test > 1:
        raise ValueError("val and test cannot be greater than 1")
    random_float = np.random.rand(len(index))
    subset_col = pd.Series(index=index, data="testing")
    subset_col[(val + test) <= random_float] = "training"
    subset_col[(test <= random_float) & (random_float < (val + test))] = "validation"
    return subset_col


def read_zip(file_path: Path) -> pd.DataFrame:
    """Reads in a zip file and returns a dataframe"""
    gpd = import_optional_dependency("geopandas")
    fiona = import_optional_dependency("fiona")
    try:
        return gpd.read_file(file_path)
    except fiona.errors.DriverError:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(file_path.parent)
        return gpd.read_file(file_path.parent / file_path.stem)


def get_lat_lon_from_centroid(geometry: pd.Series, src_crs: int = 4326):
    """Gets the lat lon from the centroid of a geometry series"""
    if (geometry == None).any():  # noqa
        raise ValueError("Geometry column cannot contain null values")
    x = geometry.centroid.x.values
    y = geometry.centroid.y.values
    if src_crs != 4326:
        t = import_optional_dependency("pyproj").Transformer
        y, x = t.from_crs(crs_from=src_crs, crs_to=4326).transform(xx=x, yy=y)
    return y, x


def sample_lat_lon_from_gdf(df: pd.DataFrame):
    """TODO: test"""
    gpd = import_optional_dependency("geopandas")
    df["samples"] = (df.geometry.area / 0.001).astype(int)

    def _get_points(polygon, samples: int):
        x_min, y_min, x_max, y_max = polygon.bounds
        x = np.random.uniform(x_min, x_max, samples)
        y = np.random.uniform(y_min, y_max, samples)
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        gdf_points = gdf_points[gdf_points.within(polygon)]
        return gdf_points

    list_of_points = np.vectorize(_get_points)(df.geometry, df.samples)
    return gpd.GeoDataFrame(geometry=pd.concat(list_of_points, ignore_index=True))
