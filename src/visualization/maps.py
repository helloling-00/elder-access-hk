# -*- coding: utf-8 -*-
"""
Map visualization functions for ELCI analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List
import colorsys

from ..config import FIGURES_DIR


# Color palettes
WALK_CMAP = mcolors.ListedColormap([
    "#084594", "#2171b5", "#4292c6",
    "#6baed6", "#bdd7e7", "#eff3ff"
])

DRIVE_CMAP = mcolors.ListedColormap([
    "#00441b", "#006d2c", "#238b45",
    "#41ab5d", "#a1d99b", "#e5f5e0"
])

TOTAL_CMAP = mcolors.ListedColormap([
    "#3f007d", "#54278f", "#756bb1",
    "#bcbddc", "#dadaeb", "#f7f7f7"
])


def setup_plot_style():
    """Set up matplotlib style for consistent plots."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: str = "YlOrRd",
    scheme: str = "Quantiles",
    k: int = 6,
    figsize: Tuple[int, int] = (12, 12),
    output_path: Optional[Union[str, Path]] = None,
    legend_label: Optional[str] = None,
    edgecolor: str = "#888888",
    linewidth: float = 0.3
) -> plt.Figure:
    """
    Create a choropleth map.

    Args:
        gdf: GeoDataFrame with geometries and data
        column: Column to visualize
        title: Plot title
        cmap: Colormap name or object
        scheme: Classification scheme
        k: Number of classes
        figsize: Figure size
        output_path: Path to save figure
        legend_label: Legend label
        edgecolor: Edge color for polygons
        linewidth: Edge line width

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    gdf.plot(
        column=column,
        cmap=cmap,
        scheme=scheme,
        k=k,
        linewidth=linewidth,
        edgecolor=edgecolor,
        legend=True,
        ax=ax
    )

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_elci_map(
    gdf: gpd.GeoDataFrame,
    elci_col: str = "ELCI_score",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot ELCI choropleth map.

    Args:
        gdf: GeoDataFrame with ELCI data
        elci_col: Column with ELCI scores
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    return plot_choropleth(
        gdf,
        column=elci_col,
        title="Elderly Living Convenience Index (TPU Level)",
        cmap="YlOrRd",
        output_path=output_path or FIGURES_DIR / "ELCI_map.png",
        legend_label="ELCI Score (0-100)"
    )


def plot_accessibility_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap=None,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot accessibility choropleth map.

    Args:
        gdf: GeoDataFrame with accessibility data
        column: Accessibility column
        title: Plot title
        cmap: Colormap (default: viridis)
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    cmap = cmap or "viridis"
    return plot_choropleth(
        gdf,
        column=column,
        title=title,
        cmap=cmap,
        output_path=output_path
    )


def plot_cluster_map(
    gdf: gpd.GeoDataFrame,
    cluster_col: str = "cluster_kmeans",
    title: str = "K-means Spatial Cluster Map",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot cluster map.

    Args:
        gdf: GeoDataFrame with cluster labels
        cluster_col: Cluster column
        title: Plot title
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 12))

    gdf.plot(
        column=cluster_col,
        cmap="Set2",
        categorical=True,
        legend=True,
        edgecolor="grey",
        linewidth=0.3,
        ax=ax
    )

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_priority_zones(
    gdf: gpd.GeoDataFrame,
    priority_col: str = "priority_zone",
    title: str = "High-Age × Low-ELCI Priority Zones",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot priority intervention zones.

    Args:
        gdf: GeoDataFrame with priority zone flags
        priority_col: Column with priority zone boolean
        title: Plot title
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 12))

    # Background (all TPUs)
    gdf.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.3)

    # Priority zones highlighted
    gdf[gdf[priority_col]].plot(
        ax=ax, color="red", edgecolor="black", linewidth=0.4
    )

    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_multimodal_accessibility(
    gdf: gpd.GeoDataFrame,
    walk_col: str,
    drive_col: str,
    total_col: str,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot multi-modal accessibility using hue-value color encoding.

    Args:
        gdf: GeoDataFrame with accessibility data
        walk_col: Walking accessibility column (normalized)
        drive_col: Driving accessibility column (normalized)
        total_col: Total score column
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    gdf = gdf.copy()

    # Min-max normalize if not already
    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    walk = minmax(gdf[walk_col].fillna(gdf[walk_col].max()))
    drive = minmax(gdf[drive_col].fillna(gdf[drive_col].max()))

    # Hue: walk (blue) → drive (green)
    hue = drive * 0.33 + walk * 0.60

    # Value: overall convenience (brighter = better)
    value = 1 - minmax(gdf[total_col].fillna(0)) * 0.8

    # Fixed saturation
    sat = 0.95

    # Convert to RGB
    colors = []
    for h, v in zip(hue, value):
        r, g, b = colorsys.hsv_to_rgb(h, sat, v)
        colors.append((r, g, b))

    gdf["color"] = colors

    fig, ax = plt.subplots(figsize=(12, 10))

    gdf.plot(
        color=gdf["color"],
        edgecolor="#444",
        linewidth=0.3,
        ax=ax
    )

    ax.set_title(
        "Multi-Modal Healthcare Accessibility (Hue-Value Mixed Rendering)",
        fontsize=16, fontweight="bold"
    )
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_side_by_side_maps(
    gdf: gpd.GeoDataFrame,
    columns: List[str],
    titles: List[str],
    cmaps: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot multiple maps side by side.

    Args:
        gdf: GeoDataFrame
        columns: List of columns to plot
        titles: List of titles for each subplot
        cmaps: List of colormaps
        output_path: Output file path

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 9))

    if n == 1:
        axes = [axes]

    cmaps = cmaps or ["OrRd"] * n

    for ax, col, title, cmap in zip(axes, columns, titles, cmaps):
        gdf.plot(
            column=col,
            cmap=cmap,
            edgecolor="gray",
            linewidth=0.3,
            legend=True,
            ax=ax
        )
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig
