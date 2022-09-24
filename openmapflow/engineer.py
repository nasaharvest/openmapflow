import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.compat._optional import import_optional_dependency

from openmapflow.bands import (
    BANDS,
    DAYS_PER_TIMESTEP,
    DYNAMIC_BANDS,
    RAW_BANDS,
    REMOVED_BANDS,
    STATIC_BANDS,
)


def _fillna(data):
    """Fill in the missing values in the data array"""
    bands_np = np.array(DYNAMIC_BANDS + STATIC_BANDS)
    if len(data.shape) != 4:
        raise ValueError(
            f"Expected data to be 4D (time, band, x, y) - got {data.shape}"
        )
    if data.shape[1] != len(bands_np):
        raise ValueError(
            f"Expected data to have {len(bands_np)} bands - got {data.shape[1]}"
        )

    is_nan = np.isnan(data)
    if not is_nan.any().item():
        return data
    mean_per_time_band = data.mean(axis=(2, 3), skipna=True)
    is_nan_any = is_nan.any(axis=(0, 2, 3)).values
    is_nan_all = is_nan.all(axis=(0, 2, 3)).values
    bands_all_nans = bands_np[is_nan_all]
    bands_some_nans = bands_np[is_nan_any & ~is_nan_all]
    if bands_all_nans.size > 0:
        print(f"WARNING: Bands: {bands_all_nans} have all nan values")
        # If a band has all nan values, fill with default: 0
        mean_per_time_band[:, is_nan_all] = 0
    if bands_some_nans.size > 0:
        print(f"WARNING: Bands: {bands_some_nans} have some nan values")
    if np.isnan(mean_per_time_band).any():
        mean_per_band = mean_per_time_band.mean(axis=0, skipna=True)
        return data.fillna(mean_per_band)
    return data.fillna(mean_per_time_band)


def load_tif(
    filepath: Path, start_date: Optional[datetime] = None, fillna: bool = True
):
    r"""
    The sentinel files exported from google earth have all the timesteps
    concatenated together. This function loads a tif files and splits the
    timesteps

    Returns: The loaded xr.DataArray
    """
    xr = import_optional_dependency("xarray")

    da = xr.open_rasterio(filepath).rename("FEATURES")

    da_split_by_time = []

    bands_per_timestep = len(DYNAMIC_BANDS)
    num_bands = len(da.band)
    num_dynamic_bands = num_bands - len(STATIC_BANDS)

    assert num_dynamic_bands % bands_per_timestep == 0
    num_timesteps = num_dynamic_bands // bands_per_timestep

    static_data = da.isel(band=slice(num_bands - len(STATIC_BANDS), num_bands))

    for timestep in range(num_timesteps):
        time_specific_da = da.isel(
            band=slice(
                timestep * bands_per_timestep, (timestep + 1) * bands_per_timestep
            )
        )
        time_specific_da = xr.concat([time_specific_da, static_data], "band")
        time_specific_da["band"] = range(bands_per_timestep + len(STATIC_BANDS))
        da_split_by_time.append(time_specific_da)

    if start_date:
        timesteps = [
            start_date + timedelta(days=DAYS_PER_TIMESTEP) * i
            for i in range(len(da_split_by_time))
        ]
        data = xr.concat(da_split_by_time, pd.Index(timesteps, name="time"))
    else:
        data = xr.concat(
            da_split_by_time, pd.Index(range(len(da_split_by_time)), name="time")
        )
    if fillna:
        data = _fillna(data)
    data.attrs["band_descriptions"] = BANDS
    return data


def calculate_ndvi(input_array: np.ndarray) -> np.ndarray:
    r"""
    Given an input array of shape [timestep, bands] or [batches, timesteps, shapes]
    where bands == len(bands), returns an array of shape
    [timestep, bands + 1] where the extra band is NDVI,
    (b08 - b04) / (b08 + b04)
    """
    band_1, band_2 = "B8", "B4"
    num_dims = len(input_array.shape)
    if num_dims == 2:
        band_1_np = input_array[:, BANDS.index(band_1)]
        band_2_np = input_array[:, BANDS.index(band_2)]
    elif num_dims == 3:
        band_1_np = input_array[:, :, BANDS.index(band_1)]
        band_2_np = input_array[:, :, BANDS.index(band_2)]
    else:
        raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in true_divide"
        )
        # suppress the following warning
        # RuntimeWarning: invalid value encountered in true_divide
        # for cases where near_infrared + red == 0
        # since this is handled in the where condition
        ndvi = np.where(
            (band_1_np + band_2_np) > 0,
            (band_1_np - band_2_np) / (band_1_np + band_2_np),
            0,
        )
    return np.append(input_array, np.expand_dims(ndvi, -1), axis=-1)


def remove_bands(array: np.ndarray) -> np.ndarray:
    """
    Expects the input to be of shape [timesteps, bands] or
    [batches, timesteps, bands]
    """
    num_dims = len(array.shape)
    error_message = f"Expected num_dims to be 2 or 3 - got {num_dims}"
    if num_dims == 2:
        bands_index = 1
    elif num_dims == 3:
        bands_index = 2
    else:
        raise ValueError(error_message)

    indices_to_remove: List[int] = []
    for band in REMOVED_BANDS:
        indices_to_remove.append(RAW_BANDS.index(band))
    indices_to_keep = [
        i for i in range(array.shape[bands_index]) if i not in indices_to_remove
    ]
    if num_dims == 2:
        return array[:, indices_to_keep]
    elif num_dims == 3:
        return array[:, :, indices_to_keep]
    else:
        # Unreachable code logically but mypy does not see it this way
        raise ValueError(error_message)


def process_test_file(path_to_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = load_tif(path_to_file)

    # Process remote sensing data
    x_np = da.values
    x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
    x_np = np.moveaxis(x_np, -1, 0)
    x_np = calculate_ndvi(x_np)
    x_np = remove_bands(x_np)

    # Get lat lons
    lon, lat = np.meshgrid(da.x.values, da.y.values)
    flat_lat, flat_lon = (
        np.squeeze(lat.reshape(-1, 1), -1),
        np.squeeze(lon.reshape(-1, 1), -1),
    )
    return x_np, flat_lat, flat_lon
