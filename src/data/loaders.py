# -*- coding: utf-8 -*-
"""
Data loading functions for various input files.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional, Union

from ..config import DATA_DIR, FILE_PATHS, CRS_HK
from ..utils.data_utils import load_geodataframe


def load_tpu_boundaries(
    filepath: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Load TPU (Tertiary Planning Unit) boundary polygons.

    Args:
        filepath: Path to TPU shapefile or geopackage

    Returns:
        GeoDataFrame with TPU boundaries
    """
    # Try multiple possible file paths
    if filepath is None:
        possible_paths = [
            DATA_DIR / FILE_PATHS["tpu_boundary"],  # Default: STPUG_21C_converted.shp
            DATA_DIR / "STPUG_21C.gpkg",
            DATA_DIR / "STPUG_21C_converted.gpkg",
            DATA_DIR / "tpu_boundary.gpkg",
        ]

        filepath = None
        for p in possible_paths:
            if p.exists():
                filepath = p
                break

        if filepath is None:
            raise FileNotFoundError(
                f"TPU boundary file not found. Searched in:\n"
                f"  {possible_paths}\n\n"
                f"Please either:\n"
                f"  1. Run: python run_pipeline.py --fetch-tpu\n"
                f"  2. Or manually download from: https://data.gov.hk/en-data/dataset/hk-censtatd-censtatd_gis-sma-boundary\n"
                f"     and place the shapefile in {DATA_DIR}/"
            )

    print(f"Loading TPU boundaries from {filepath}...")
    tpu = load_geodataframe(filepath, target_crs=CRS_HK)
    print(f"Loaded {len(tpu)} TPU polygons")

    return tpu


def load_residential_buildings(
    filepath: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Load residential building centroids.

    Args:
        filepath: Path to residential buildings file

    Returns:
        GeoDataFrame with residential building centroids
    """
    filepath = filepath or DATA_DIR / FILE_PATHS["residential"]
    print(f"Loading residential buildings from {filepath}...")

    resi = load_geodataframe(filepath, target_crs=CRS_HK)

    # Ensure we have centroids
    if "centroid" not in resi.columns:
        resi["centroid"] = resi.geometry.centroid

    print(f"Loaded {len(resi)} residential buildings")

    return resi


def load_transit_pois(
    filepath: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Load transit POIs.

    Args:
        filepath: Path to transit file

    Returns:
        GeoDataFrame with transit POIs
    """
    filepath = filepath or DATA_DIR / FILE_PATHS["transit"]
    return load_geodataframe(filepath, target_crs=CRS_HK)


def load_healthcare_pois(
    filepath: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Load healthcare POIs.

    Args:
        filepath: Path to healthcare file

    Returns:
        GeoDataFrame with healthcare POIs
    """
    filepath = filepath or DATA_DIR / FILE_PATHS["healthcare"]
    return load_geodataframe(filepath, target_crs=CRS_HK)


def load_daily_shopping_pois(
    filepath: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Load daily shopping POIs.

    Args:
        filepath: Path to daily shopping file

    Returns:
        GeoDataFrame with daily shopping POIs
    """
    filepath = filepath or DATA_DIR / FILE_PATHS["daily_shopping"]
    return load_geodataframe(filepath, target_crs=CRS_HK)


def load_convenience_service_pois(
    filepath: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Load convenience service POIs.

    Args:
        filepath: Path to convenience services file

    Returns:
        GeoDataFrame with convenience service POIs
    """
    filepath = filepath or DATA_DIR / FILE_PATHS["convenience_services"]
    return load_geodataframe(filepath, target_crs=CRS_HK)


def load_leisure_recreation_pois(
    filepath: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Load leisure/recreation POIs.

    Args:
        filepath: Path to leisure file

    Returns:
        GeoDataFrame with leisure/recreation POIs
    """
    filepath = filepath or DATA_DIR / FILE_PATHS["leisure_recreation"]
    return load_geodataframe(filepath, target_crs=CRS_HK)


def load_population_data(
    filepath: Union[str, Path],
    tpu_col: str = "TPU_2021"
) -> pd.DataFrame:
    """
    Load elderly population data.

    Args:
        filepath: Path to population Excel file
        tpu_col: Column name for TPU identifier

    Returns:
        DataFrame with population data
    """
    print(f"Loading population data from {filepath}...")
    df = pd.read_excel(filepath)
    df[tpu_col] = df[tpu_col].astype(str)
    print(f"Loaded {len(df)} records")
    return df


def load_elci_results(
    filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load ELCI calculation results.

    Args:
        filepath: Path to ELCI results Excel file

    Returns:
        DataFrame with ELCI results
    """
    from ..config import OUTPUT_FILES, OUTPUT_DIR

    filepath = filepath or OUTPUT_DIR / OUTPUT_FILES["elci_result"]
    print(f"Loading ELCI results from {filepath}...")

    df = pd.read_excel(filepath)
    print(f"Loaded {len(df)} TPU records")

    return df
