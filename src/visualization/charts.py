# -*- coding: utf-8 -*-
"""
Chart visualization functions for ELCI analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from ..config import FIGURES_DIR


def setup_plot_style():
    """Set up matplotlib style for consistent plots."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12
    sns.set_theme(style="whitegrid")


def plot_elbow_curve(
    inertias: List[float],
    k_range: range,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot elbow curve for optimal k selection.

    Args:
        inertias: List of inertia values
        k_range: Range of k values tested
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(list(k_range), inertias, "-o", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title("Elbow Method for Optimal k", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_cluster_heatmap(
    cluster_profile: pd.DataFrame,
    title: str = "Cluster Profiles (Normalized Scores)",
    pretty_names: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot cluster profile heatmap.

    Args:
        cluster_profile: DataFrame with cluster profiles
        title: Plot title
        pretty_names: Dict mapping column names to display names
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    if pretty_names:
        cluster_profile = cluster_profile.rename(columns=pretty_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        cluster_profile,
        annot=True,
        cmap="YlGnBu",
        linewidths=0.3,
        annot_kws={"size": 10, "color": "black"},
        ax=ax
    )

    ax.set_title(title, fontsize=16, pad=15, fontweight="bold")
    ax.set_xlabel("Accessibility Dimensions", fontsize=13)
    ax.set_ylabel("Cluster", fontsize=13)

    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=11)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_scatter_population_elci(
    df: pd.DataFrame,
    population_col: str,
    elci_col: str = "ELCI_score",
    hue_col: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot scatter plot of elderly population vs ELCI.

    Args:
        df: DataFrame with population and ELCI data
        population_col: Column with population counts
        elci_col: Column with ELCI scores
        hue_col: Column for color encoding (optional)
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    hue_col = hue_col or elci_col

    sns.scatterplot(
        data=df,
        x=population_col,
        y=elci_col,
        hue=hue_col,
        palette="coolwarm",
        alpha=0.7,
        s=60,
        ax=ax
    )

    ax.set_xlabel("Elderly Population", fontsize=12)
    ax.set_ylabel("Accessibility (ELCI Score)", fontsize=12)
    ax.set_title("Relationship: Elderly Population vs Accessibility", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_regression_results(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot actual vs predicted scatter plot.

    Args:
        actual: Actual values
        predicted: Predicted values
        model_name: Name of the model
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual, predicted, alpha=0.5, s=40)

    # Add perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(f"{model_name}: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    title: str = "Hierarchical Clustering Dendrogram",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot hierarchical clustering dendrogram.

    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        title: Plot title
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    from scipy.cluster.hierarchy import dendrogram

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=0.7 * max(linkage_matrix[:, 2]),
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_feature_importance(
    importance_dict: Dict[str, float],
    title: str = "Feature Importance",
    pretty_names: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot feature importance bar chart.

    Args:
        importance_dict: Dict mapping feature names to importance scores
        title: Plot title
        pretty_names: Dict mapping feature names to display names
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    if pretty_names:
        importance_dict = {
            pretty_names.get(k, k): v
            for k, v in importance_dict.items()
        }

    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_items]
    importances = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(features, importances, color="steelblue")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, importances):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.invert_yaxis()  # Highest importance at top
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    bins: int = 30,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot histogram distribution.

    Args:
        df: Input DataFrame
        column: Column to plot
        title: Plot title
        bins: Number of bins
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df[column].dropna(), bins=bins, color="steelblue", edgecolor="white")
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title or f"Distribution of {column}", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig
