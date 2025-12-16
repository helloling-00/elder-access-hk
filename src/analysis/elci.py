# -*- coding: utf-8 -*-
"""
ELCI (Elderly Living Convenience Index) calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import ELCI_WEIGHTS
from ..utils.data_utils import minmax_normalize, scale_to_100


# Default dimension columns (accessibility times - lower is better)
DEFAULT_DIMENSIONS = [
    "transit_walk_min",
    "hospital_walk_min",
    "hospital_drive_min",
    "daily_walk_min",
    "service_walk_min",
    "leisure_walk_min",
]


def calculate_elci(
    df: pd.DataFrame,
    dimension_cols: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    output_raw_col: str = "ELCI_raw",
    output_score_col: str = "ELCI_score"
) -> pd.DataFrame:
    """
    Calculate ELCI (Elderly Living Convenience Index) from accessibility dimensions.

    The ELCI is computed by:
    1. Normalizing each dimension (reverse, since lower time = better)
    2. Computing weighted average
    3. Scaling to 0-100 range

    Args:
        df: DataFrame with accessibility columns
        dimension_cols: List of columns to use (default: 6 standard dimensions)
        weights: Dict mapping dimension names to weights (default: equal weights)
        output_raw_col: Column name for raw ELCI (0-1)
        output_score_col: Column name for scaled ELCI (0-100)

    Returns:
        DataFrame with ELCI columns added
    """
    df = df.copy()
    dimension_cols = dimension_cols or DEFAULT_DIMENSIONS
    weights = weights or {col: 1.0 for col in dimension_cols}

    # Normalize each dimension (reverse since lower time = better)
    norm_cols = []
    for col in dimension_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found, skipping")
            continue

        norm_col = f"{col}_norm"
        df[norm_col] = minmax_normalize(df[col], reverse=True)
        norm_cols.append((norm_col, weights.get(col, 1.0)))

    if not norm_cols:
        raise ValueError("No valid dimension columns found")

    # Compute weighted average
    total_weight = sum(w for _, w in norm_cols)
    weighted_sum = sum(df[col] * w for col, w in norm_cols)
    df[output_raw_col] = weighted_sum / total_weight

    # Scale to 0-100
    df[output_score_col] = scale_to_100(df[output_raw_col])

    print(f"ELCI calculated for {len(df)} TPUs")
    print(f"Score range: {df[output_score_col].min():.1f} - {df[output_score_col].max():.1f}")

    return df


def categorize_elci(
    df: pd.DataFrame,
    score_col: str = "ELCI_score",
    output_col: str = "ELCI_category",
    thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Categorize ELCI scores into levels.

    Args:
        df: DataFrame with ELCI scores
        score_col: Column containing ELCI scores
        output_col: Output column for categories
        thresholds: Dict with category thresholds (default: High/Medium/Low)

    Returns:
        DataFrame with category column added
    """
    df = df.copy()

    if thresholds is None:
        thresholds = {
            "High": 70,
            "Medium": 40,
        }

    def categorize(score):
        if pd.isna(score):
            return "Unknown"
        if score >= thresholds.get("High", 70):
            return "High"
        elif score >= thresholds.get("Medium", 40):
            return "Medium"
        else:
            return "Low"

    df[output_col] = df[score_col].apply(categorize)

    # Print distribution
    print("\nELCI Category Distribution:")
    print(df[output_col].value_counts())

    return df


class ELCICalculator:
    """
    Class to manage ELCI calculation workflow.
    """

    def __init__(
        self,
        dimension_cols: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ELCI calculator.

        Args:
            dimension_cols: Columns to use for calculation
            weights: Dimension weights
        """
        self.dimension_cols = dimension_cols or DEFAULT_DIMENSIONS
        self.weights = weights or {col: 1.0 for col in self.dimension_cols}
        self.result: Optional[pd.DataFrame] = None

    def calculate(
        self,
        accessibility_df: pd.DataFrame,
        tpu_id_col: str = "stpug_eng"
    ) -> pd.DataFrame:
        """
        Calculate ELCI from accessibility data.

        Args:
            accessibility_df: DataFrame with TPU-level accessibility
            tpu_id_col: TPU identifier column

        Returns:
            DataFrame with ELCI results
        """
        self.result = calculate_elci(
            accessibility_df,
            self.dimension_cols,
            self.weights
        )
        return self.result

    def categorize(
        self,
        thresholds: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Add category column to results.

        Args:
            thresholds: Category thresholds

        Returns:
            DataFrame with categories
        """
        if self.result is None:
            raise ValueError("Must calculate ELCI first")

        self.result = categorize_elci(self.result, thresholds=thresholds)
        return self.result

    def save(
        self,
        output_path: Union[str, Path],
        include_normalized: bool = True
    ) -> None:
        """
        Save ELCI results to Excel.

        Args:
            output_path: Output file path
            include_normalized: Whether to include normalized columns
        """
        if self.result is None:
            raise ValueError("Must calculate ELCI first")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.result
        if not include_normalized:
            # Drop normalized columns
            df = df[[c for c in df.columns if not c.endswith("_norm")]]

        df.to_excel(output_path, index=False)
        print(f"Saved ELCI results to {output_path}")
