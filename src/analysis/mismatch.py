# -*- coding: utf-8 -*-
"""
Mismatch analysis between elderly population and accessibility.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Optional, Tuple, Union
from pathlib import Path


def identify_priority_zones(
    df: pd.DataFrame,
    population_col: str,
    elci_col: str = "ELCI_score",
    pop_quantile: float = 0.75,
    elci_quantile: float = 0.25,
    output_col: str = "priority_zone"
) -> pd.DataFrame:
    """
    Identify high-priority zones with high elderly population but low accessibility.

    Args:
        df: DataFrame with population and ELCI data
        population_col: Column with elderly population counts
        elci_col: Column with ELCI scores
        pop_quantile: Quantile threshold for "high" population
        elci_quantile: Quantile threshold for "low" ELCI
        output_col: Output column name

    Returns:
        DataFrame with priority zone flags
    """
    df = df.copy()

    # Calculate thresholds
    pop_threshold = df[population_col].quantile(pop_quantile)
    elci_threshold = df[elci_col].quantile(elci_quantile)

    print(f"Population threshold (top {(1-pop_quantile)*100:.0f}%): {pop_threshold:.0f}")
    print(f"ELCI threshold (bottom {elci_quantile*100:.0f}%): {elci_threshold:.1f}")

    # Identify high-population, low-ELCI zones
    df["high_elderly"] = df[population_col] > pop_threshold
    df["low_elci"] = df[elci_col] < elci_threshold
    df[output_col] = df["high_elderly"] & df["low_elci"]

    n_priority = df[output_col].sum()
    print(f"\nPriority zones identified: {n_priority}")

    return df


def compute_mismatch_score(
    df: pd.DataFrame,
    population_col: str,
    elci_col: str = "ELCI_score",
    output_col: str = "mismatch_score"
) -> pd.DataFrame:
    """
    Compute a mismatch score combining population and accessibility.

    Higher score = more elderly + worse accessibility = higher priority for intervention.

    Args:
        df: DataFrame with population and ELCI data
        population_col: Column with elderly population
        elci_col: Column with ELCI scores
        output_col: Output column name

    Returns:
        DataFrame with mismatch scores
    """
    df = df.copy()

    # Normalize population (higher = more elderly)
    pop_norm = (df[population_col] - df[population_col].min()) / \
               (df[population_col].max() - df[population_col].min())

    # Invert ELCI (lower ELCI = higher need)
    elci_inv = 1 - (df[elci_col] - df[elci_col].min()) / \
               (df[elci_col].max() - df[elci_col].min())

    # Combine (equal weights)
    df[output_col] = (pop_norm + elci_inv) / 2

    print(f"Mismatch score range: {df[output_col].min():.3f} - {df[output_col].max():.3f}")

    return df


def rank_priority_areas(
    df: pd.DataFrame,
    mismatch_col: str = "mismatch_score",
    tpu_id_col: str = "stpug_eng",
    top_n: int = 20
) -> pd.DataFrame:
    """
    Rank TPUs by priority for intervention.

    Args:
        df: DataFrame with mismatch scores
        mismatch_col: Column with mismatch scores
        tpu_id_col: TPU identifier column
        top_n: Number of top priority areas to return

    Returns:
        Top priority areas sorted by mismatch score
    """
    priority = (
        df[[tpu_id_col, mismatch_col]]
        .sort_values(mismatch_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    priority["rank"] = range(1, len(priority) + 1)

    print(f"\nTop {top_n} Priority Areas:")
    print(priority)

    return priority


class MismatchAnalyzer:
    """
    Class to manage mismatch analysis workflow.
    """

    def __init__(
        self,
        tpu_id_col: str = "stpug_eng",
        elci_col: str = "ELCI_score"
    ):
        """
        Initialize mismatch analyzer.

        Args:
            tpu_id_col: TPU identifier column
            elci_col: ELCI score column
        """
        self.tpu_id_col = tpu_id_col
        self.elci_col = elci_col
        self.result: Optional[pd.DataFrame] = None

    def analyze(
        self,
        elci_df: pd.DataFrame,
        population_df: pd.DataFrame,
        population_col: str,
        merge_on: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Perform mismatch analysis.

        Args:
            elci_df: DataFrame with ELCI results
            population_df: DataFrame with population data
            population_col: Column with elderly population
            merge_on: Column to merge on (defaults to tpu_id_col)

        Returns:
            DataFrame with mismatch analysis results
        """
        merge_on = merge_on or self.tpu_id_col

        # Merge data
        df = elci_df.merge(
            population_df[[merge_on, population_col]],
            on=merge_on,
            how="left"
        )

        # Compute mismatch score
        df = compute_mismatch_score(df, population_col, self.elci_col)

        # Identify priority zones
        df = identify_priority_zones(df, population_col, self.elci_col)

        self.result = df
        return df

    def get_priority_ranking(
        self,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get ranked list of priority areas.

        Args:
            top_n: Number of top areas

        Returns:
            Ranked priority areas
        """
        if self.result is None:
            raise ValueError("Must run analyze() first")

        return rank_priority_areas(
            self.result, "mismatch_score",
            self.tpu_id_col, top_n
        )

    def get_priority_zones_gdf(
        self,
        tpu_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Get GeoDataFrame of priority zones.

        Args:
            tpu_gdf: TPU boundary GeoDataFrame

        Returns:
            GeoDataFrame with priority zone geometries
        """
        if self.result is None:
            raise ValueError("Must run analyze() first")

        merged = tpu_gdf.merge(
            self.result[[self.tpu_id_col, "priority_zone", "mismatch_score"]],
            on=self.tpu_id_col,
            how="left"
        )

        return merged

    def save(
        self,
        output_path: Union[str, Path]
    ) -> None:
        """
        Save mismatch analysis results.

        Args:
            output_path: Output file path
        """
        if self.result is None:
            raise ValueError("Must run analyze() first")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.result.to_excel(output_path, index=False)
        print(f"Saved mismatch analysis to {output_path}")
