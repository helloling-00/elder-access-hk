# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities.
"""

import re
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

from ..config import CRS_HK, CRS_WGS84


def load_geodataframe(
    filepath: Union[str, Path],
    target_crs: str = CRS_HK,
    layer: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from various formats and reproject to target CRS.

    Args:
        filepath: Path to the file (gpkg, shp, geojson, etc.)
        target_crs: Target coordinate reference system
        layer: Layer name for multi-layer files (e.g., gpkg)

    Returns:
        GeoDataFrame in target CRS
    """
    filepath = Path(filepath)

    if layer:
        gdf = gpd.read_file(filepath, layer=layer)
    else:
        gdf = gpd.read_file(filepath)

    if gdf.crs is None:
        print(f"Warning: No CRS found for {filepath}, assuming {CRS_WGS84}")
        gdf = gdf.set_crs(CRS_WGS84)

    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def save_geodataframe(
    gdf: gpd.GeoDataFrame,
    filepath: Union[str, Path],
    driver: str = "GPKG",
    layer: Optional[str] = None
) -> None:
    """
    Save a GeoDataFrame to file.

    Args:
        gdf: GeoDataFrame to save
        filepath: Output path
        driver: Output format driver
        layer: Layer name for multi-layer formats
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if layer:
        gdf.to_file(filepath, driver=driver, layer=layer)
    else:
        gdf.to_file(filepath, driver=driver)

    print(f"Saved: {filepath}")


def sanitize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean column names to be valid for file formats like GPKG/Shapefile.

    - Replace illegal characters with underscore
    - Avoid empty names
    - Avoid names starting with digits
    - Ensure unique names

    Args:
        gdf: Input GeoDataFrame

    Returns:
        GeoDataFrame with sanitized column names
    """
    new_cols = []
    seen = set()

    for col in gdf.columns:
        # Replace illegal characters
        safe = re.sub(r'[^0-9a-zA-Z_]', '_', str(col))

        # Avoid empty names
        if safe == "":
            safe = "field"

        # Avoid names starting with digit
        if safe[0].isdigit():
            safe = "_" + safe

        # Ensure uniqueness
        base = safe
        i = 1
        while safe in seen:
            safe = f"{base}_{i}"
            i += 1

        seen.add(safe)
        new_cols.append(safe)

    rename_map = {old: new for old, new in zip(gdf.columns, new_cols)}
    return gdf.rename(columns=rename_map)


def load_excel_with_geometry(
    filepath: Union[str, Path],
    geometry_col: str = "geometry",
    crs: str = CRS_WGS84,
    target_crs: str = CRS_HK
) -> gpd.GeoDataFrame:
    """
    Load Excel file with WKT geometry column and convert to GeoDataFrame.

    Args:
        filepath: Path to Excel file
        geometry_col: Name of column containing WKT geometry
        crs: Source CRS
        target_crs: Target CRS

    Returns:
        GeoDataFrame
    """
    from shapely import wkt
    from shapely.geometry import Point

    df = pd.read_excel(filepath)

    def parse_geometry(s):
        try:
            if isinstance(s, str):
                # Handle POINT format
                if s.startswith("POINT"):
                    nums = s.replace("POINT", "").replace("(", "").replace(")", "").split()
                    return Point(float(nums[0]), float(nums[1]))
                return wkt.loads(s)
        except Exception:
            pass
        return None

    df["geometry"] = df[geometry_col].apply(parse_geometry)
    df = df[df["geometry"].notna()]

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    return gdf.to_crs(target_crs)


def minmax_normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    Min-max normalization.

    Args:
        series: Input series
        reverse: If True, reverse the normalization (for "lower is better" metrics)

    Returns:
        Normalized series (0-1 range)
    """
    s_min, s_max = series.min(), series.max()

    if s_max == s_min:
        return pd.Series([0.5] * len(series), index=series.index)

    if reverse:
        return (s_max - series) / (s_max - s_min)
    else:
        return (series - s_min) / (s_max - s_min)


def scale_to_100(series: pd.Series) -> pd.Series:
    """
    Scale a series to 0-100 range.

    Args:
        series: Input series (typically 0-1 range)

    Returns:
        Scaled series (0-100 range)
    """
    s_min, s_max = series.min(), series.max()

    if s_max == s_min:
        return pd.Series([50] * len(series), index=series.index)

    return 100 * (series - s_min) / (s_max - s_min)
