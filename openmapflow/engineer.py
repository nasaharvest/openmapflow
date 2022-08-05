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


def load_tif(
    filepath: Path,
    start_date: datetime,
    num_timesteps: Optional[int] = None,
):
    r"""
    The sentinel files exported from google earth have all the timesteps
    concatenated together. This function loads a tif files and splits the
    timesteps

    Returns: The loaded xr.DataArray, and the average slope (used for filling nan slopes)
    """
    xr = import_optional_dependency("xarray")

    da = xr.open_rasterio(filepath).rename("FEATURES")

    da_split_by_time = []

    bands_per_timestep = len(DYNAMIC_BANDS)
    num_bands = len(da.band)
    num_dynamic_bands = num_bands - len(STATIC_BANDS)

    assert num_dynamic_bands % bands_per_timestep == 0
    if num_timesteps is None:
        num_timesteps = num_dynamic_bands // bands_per_timestep

    static_data = da.isel(band=slice(num_bands - len(STATIC_BANDS), num_bands))
    average_slope = np.nanmean(static_data.values[STATIC_BANDS.index("slope"), :, :])

    for timestep in range(num_timesteps):
        time_specific_da = da.isel(
            band=slice(
                timestep * bands_per_timestep, (timestep + 1) * bands_per_timestep
            )
        )
        time_specific_da = xr.concat([time_specific_da, static_data], "band")
        time_specific_da["band"] = range(bands_per_timestep + len(STATIC_BANDS))
        da_split_by_time.append(time_specific_da)

    timesteps = [
        start_date + timedelta(days=DAYS_PER_TIMESTEP) * i
        for i in range(len(da_split_by_time))
    ]

    dynamic_data = xr.concat(da_split_by_time, pd.Index(timesteps, name="time"))
    dynamic_data.attrs["band_descriptions"] = BANDS

    return dynamic_data, average_slope


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


def fillna(array: np.ndarray, average_slope: float) -> Optional[np.ndarray]:
    r"""
    Given an input array of shape [timesteps, BANDS]
    fill NaN values with the mean of each band across the timestep
    The slope values may all be nan so average_slope is manually passed
    """
    num_dims = len(array.shape)
    if num_dims == 2:
        bands_index = 1
        mean_per_band = np.nanmean(array, axis=0)
    elif num_dims == 3:
        bands_index = 2
        mean_per_band = np.nanmean(np.nanmean(array, axis=0), axis=0)
    else:
        raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

    assert array.shape[bands_index] == len(BANDS)

    if np.isnan(mean_per_band).any():
        if (sum(np.isnan(mean_per_band)) == bands_index) & (
            np.isnan(mean_per_band[BANDS.index("slope")]).all()
        ):
            mean_per_band[BANDS.index("slope")] = average_slope
            assert not np.isnan(mean_per_band).any()
        else:
            return None
    for i in range(array.shape[bands_index]):
        if num_dims == 2:
            array[:, i] = np.nan_to_num(array[:, i], nan=mean_per_band[i])
        elif num_dims == 3:
            array[:, :, i] = np.nan_to_num(array[:, :, i], nan=mean_per_band[i])
    return array


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


def process_test_file(
    path_to_file: Path, start_date: datetime, num_timesteps: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    da, slope = load_tif(
        path_to_file, start_date=start_date, num_timesteps=num_timesteps
    )

    # Process remote sensing data
    x_np = da.values
    x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
    x_np = np.moveaxis(x_np, -1, 0)
    x_np = calculate_ndvi(x_np)
    x_np = remove_bands(x_np)
    final_x = fillna(x_np, slope)
    if final_x is None:
        raise RuntimeError(
            "fillna on the test instance returned None; "
            "does the test instance contain NaN only bands?"
        )

    # Get lat lons
    lon, lat = np.meshgrid(da.x.values, da.y.values)
    flat_lat, flat_lon = (
        np.squeeze(lat.reshape(-1, 1), -1),
        np.squeeze(lon.reshape(-1, 1), -1),
    )
    return final_x, flat_lat, flat_lon
