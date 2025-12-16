# -*- coding: utf-8 -*-
"""
TPU-level accessibility aggregation.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional, Union, Dict, List

from ..utils.geo_utils import assign_points_to_polygons


def assign_residential_to_tpu(
    residential: gpd.GeoDataFrame,
    tpu: gpd.GeoDataFrame,
    tpu_id_col: str = "stpug_eng",
    centroid_col: str = "centroid"
) -> gpd.GeoDataFrame:
    """
    Assign residential buildings to TPU polygons.

    Args:
        residential: GeoDataFrame with residential buildings
        tpu: GeoDataFrame with TPU polygons
        tpu_id_col: Column in TPU containing IDs
        centroid_col: Column in residential containing centroids

    Returns:
        Residential GeoDataFrame with TPU assignments
    """
    residential = residential.copy()

    # Ensure centroid column exists
    if centroid_col not in residential.columns:
        residential[centroid_col] = residential.geometry.centroid

    print("Assigning residential buildings to TPU...")
    assigned = assign_points_to_polygons(
        residential, tpu, tpu_id_col, centroid_col
    )

    residential[tpu_id_col] = assigned

    matched = residential[tpu_id_col].notna().sum()
    print(f"Matched {matched}/{len(residential)} buildings to TPU")

    return residential


def aggregate_accessibility_to_tpu(
    residential: gpd.GeoDataFrame,
    accessibility_col: str,
    tpu_id_col: str = "stpug_eng",
    output_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate building-level accessibility to TPU level.

    Args:
        residential: GeoDataFrame with accessibility values
        accessibility_col: Column containing accessibility times
        tpu_id_col: Column containing TPU IDs
        output_col: Output column name (defaults to accessibility_col)

    Returns:
        DataFrame with TPU-level average accessibility
    """
    output_col = output_col or accessibility_col

    # Filter to buildings with TPU assignment
    valid = residential[residential[tpu_id_col].notna()]

    result = (
        valid.groupby(tpu_id_col)[accessibility_col]
        .mean()
        .reset_index(name=output_col)
    )

    print(f"Aggregated to {len(result)} TPUs")

    return result


def compute_unreachable_ratio(
    residential: gpd.GeoDataFrame,
    accessibility_col: str,
    tpu_id_col: str = "stpug_eng",
    unreachable_threshold: float = 60.0
) -> pd.DataFrame:
    """
    Compute ratio of unreachable buildings per TPU.

    Args:
        residential: GeoDataFrame with accessibility values
        accessibility_col: Column containing accessibility times
        tpu_id_col: Column containing TPU IDs
        unreachable_threshold: Time threshold for "unreachable"

    Returns:
        DataFrame with TPU-level unreachable ratios
    """
    residential = residential.copy()
    residential["unreachable"] = (
        residential[accessibility_col].isna() |
        (residential[accessibility_col] >= unreachable_threshold)
    )

    valid = residential[residential[tpu_id_col].notna()]

    result = (
        valid.groupby(tpu_id_col)["unreachable"]
        .mean()
        .reset_index(name="unreachable_ratio")
    )

    return result


class TPUAggregator:
    """
    Class to manage TPU-level aggregation of accessibility metrics.
    """

    def __init__(
        self,
        tpu: gpd.GeoDataFrame,
        tpu_id_col: str = "stpug_eng"
    ):
        """
        Initialize aggregator with TPU boundaries.

        Args:
            tpu: GeoDataFrame with TPU polygons
            tpu_id_col: Column containing TPU IDs
        """
        self.tpu = tpu
        self.tpu_id_col = tpu_id_col
        self._results: Dict[str, pd.DataFrame] = {}

    def add_accessibility(
        self,
        residential: gpd.GeoDataFrame,
        accessibility_col: str,
        output_name: str
    ) -> pd.DataFrame:
        """
        Add an accessibility metric to aggregation.

        Args:
            residential: GeoDataFrame with accessibility values
            accessibility_col: Column containing accessibility times
            output_name: Name for the output column

        Returns:
            TPU-level aggregated DataFrame
        """
        # Ensure TPU assignment
        if self.tpu_id_col not in residential.columns:
            residential = assign_residential_to_tpu(
                residential, self.tpu, self.tpu_id_col
            )

        result = aggregate_accessibility_to_tpu(
            residential, accessibility_col,
            self.tpu_id_col, output_name
        )

        self._results[output_name] = result
        return result

    def get_combined_results(self) -> pd.DataFrame:
        """
        Get all accessibility results combined into one DataFrame.

        Returns:
            Combined DataFrame with all metrics
        """
        if not self._results:
            return pd.DataFrame()

        # Start with first result
        combined = list(self._results.values())[0]

        # Merge remaining results
        for name, df in list(self._results.items())[1:]:
            combined = combined.merge(df, on=self.tpu_id_col, how="outer")

        return combined

    def save_results(
        self,
        output_path: Union[str, Path],
        combined: bool = True
    ) -> None:
        """
        Save aggregation results to Excel.

        Args:
            output_path: Output file path
            combined: If True, save combined results; else save individual files
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if combined:
            df = self.get_combined_results()
            df.to_excel(output_path, index=False)
            print(f"Saved combined results to {output_path}")
        else:
            for name, df in self._results.items():
                path = output_path.parent / f"{output_path.stem}_{name}.xlsx"
                df.to_excel(path, index=False)
                print(f"Saved {name} to {path}")
