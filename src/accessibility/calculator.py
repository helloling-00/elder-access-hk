# -*- coding: utf-8 -*-
"""
Accessibility calculation using network analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import List, Dict, Optional, Union, Any
from tqdm import tqdm

from ..config import MAX_WALK_TIME_MIN, MAX_DRIVE_DIST_M
from ..utils.geo_utils import SpatialIndex


def compute_multi_source_dijkstra(
    G: nx.MultiDiGraph,
    sources: List[Any],
    weight: str = "travel_time"
) -> Dict[Any, float]:
    """
    Compute shortest path lengths from multiple sources.

    Args:
        G: NetworkX graph
        sources: List of source node IDs
        weight: Edge weight attribute to use

    Returns:
        Dictionary mapping node IDs to shortest path lengths
    """
    print(f"Running multi-source Dijkstra with {len(sources)} sources...")
    lengths = nx.multi_source_dijkstra_path_length(G, sources, weight=weight)
    print("Dijkstra completed")
    return lengths


def snap_pois_to_network(
    pois: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    node_id_col: str = "osmid",
    max_dist: Optional[float] = None
) -> List[Optional[Any]]:
    """
    Snap POI locations to nearest network nodes.

    Args:
        pois: GeoDataFrame with POI locations
        nodes: GeoDataFrame with network nodes
        node_id_col: Column containing node IDs
        max_dist: Maximum snapping distance

    Returns:
        List of node IDs (None for POIs beyond max_dist)
    """
    print("Building spatial index for network nodes...")
    spatial_idx = SpatialIndex(nodes, id_column=node_id_col)

    print("Snapping POIs to network...")
    snapped = spatial_idx.snap_geodataframe(pois, max_dist=max_dist)

    valid = sum(1 for x in snapped if x is not None)
    print(f"Snapped {valid}/{len(pois)} POIs to network")

    return snapped


def calculate_accessibility(
    residential: gpd.GeoDataFrame,
    poi_sources: List[Any],
    G: nx.MultiDiGraph,
    nodes: gpd.GeoDataFrame,
    node_id_col: str = "osmid",
    max_time_min: float = MAX_WALK_TIME_MIN,
    weight: str = "travel_time"
) -> np.ndarray:
    """
    Calculate accessibility from residential buildings to POIs.

    Args:
        residential: GeoDataFrame with residential building centroids
        poi_sources: List of source node IDs (POI locations)
        G: NetworkX graph
        nodes: GeoDataFrame with network nodes
        node_id_col: Column containing node IDs
        max_time_min: Maximum travel time in minutes
        weight: Edge weight attribute

    Returns:
        Array of accessibility times in minutes for each residential building
    """
    # Compute shortest paths from POIs
    path_lengths = compute_multi_source_dijkstra(G, poi_sources, weight)

    # Build spatial index for snapping
    spatial_idx = SpatialIndex(nodes, id_column=node_id_col)

    # Get residential centroids
    if "centroid" in residential.columns:
        centroids = residential["centroid"]
    else:
        centroids = residential.geometry.centroid

    # Calculate time for each residential building
    print("Calculating accessibility for residential buildings...")
    times = []

    for pt in tqdm(centroids, desc="Processing buildings"):
        node_id = spatial_idx.nearest_from_geometry(pt)

        if node_id is None or node_id not in path_lengths:
            times.append(max_time_min)
        else:
            time_sec = path_lengths[node_id]
            time_min = time_sec / 60.0
            times.append(min(time_min, max_time_min))

    return np.array(times)


def calculate_walk_accessibility(
    residential: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    nodes: gpd.GeoDataFrame,
    node_id_col: str = "osmid",
    max_time_min: float = MAX_WALK_TIME_MIN
) -> np.ndarray:
    """
    Calculate walking accessibility from residential to POIs.

    Args:
        residential: GeoDataFrame with residential buildings
        pois: GeoDataFrame with POI locations
        G: NetworkX walking graph
        nodes: GeoDataFrame with network nodes
        node_id_col: Node ID column
        max_time_min: Maximum walking time

    Returns:
        Array of walking times in minutes
    """
    # Snap POIs to network
    pois = pois.copy()
    pois["node"] = snap_pois_to_network(pois, nodes, node_id_col)
    pois = pois[pois["node"].notna()]

    sources = pois["node"].unique().tolist()

    return calculate_accessibility(
        residential, sources, G, nodes,
        node_id_col, max_time_min, "travel_time"
    )


def calculate_drive_accessibility(
    residential: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    nodes: gpd.GeoDataFrame,
    node_id_col: str = "osmid",
    max_time_min: float = MAX_WALK_TIME_MIN,
    max_snap_dist: float = MAX_DRIVE_DIST_M
) -> np.ndarray:
    """
    Calculate driving accessibility from residential to POIs.

    Args:
        residential: GeoDataFrame with residential buildings
        pois: GeoDataFrame with POI locations
        G: NetworkX driving graph
        nodes: GeoDataFrame with network nodes
        node_id_col: Node ID column
        max_time_min: Maximum driving time
        max_snap_dist: Maximum distance for snapping to network

    Returns:
        Array of driving times in minutes
    """
    # Snap POIs to network with distance threshold
    pois = pois.copy()
    pois["node"] = snap_pois_to_network(pois, nodes, node_id_col, max_snap_dist)
    pois = pois[pois["node"].notna()]

    sources = pois["node"].unique().tolist()

    # Snap residential to network
    spatial_idx = SpatialIndex(nodes, id_column=node_id_col)

    if "centroid" in residential.columns:
        centroids = residential["centroid"]
    else:
        centroids = residential.geometry.centroid

    # Compute shortest paths
    path_lengths = compute_multi_source_dijkstra(G, sources, "travel_time")

    # Calculate times
    times = []
    for pt in tqdm(centroids, desc="Processing buildings"):
        node_id = spatial_idx.nearest_from_geometry(pt, max_snap_dist)

        if node_id is None:
            times.append(np.nan)  # Not reachable by road
        elif node_id not in path_lengths:
            times.append(np.nan)
        else:
            time_min = path_lengths[node_id] / 60.0
            times.append(min(time_min, max_time_min))

    result = np.array(times)

    # Apply minimum drive time (1 minute)
    result = np.where(result < 1.0, 1.0, result)

    return result


class AccessibilityCalculator:
    """
    Class to manage accessibility calculations for multiple POI categories.
    """

    def __init__(
        self,
        walk_graph: nx.MultiDiGraph,
        walk_nodes: gpd.GeoDataFrame,
        drive_graph: Optional[nx.MultiDiGraph] = None,
        drive_nodes: Optional[gpd.GeoDataFrame] = None,
        node_id_col: str = "osmid"
    ):
        """
        Initialize calculator with network graphs.

        Args:
            walk_graph: Walking network graph
            walk_nodes: Walking network nodes
            drive_graph: Driving network graph (optional)
            drive_nodes: Driving network nodes (optional)
            node_id_col: Column containing node IDs
        """
        self.walk_graph = walk_graph
        self.walk_nodes = walk_nodes
        self.drive_graph = drive_graph
        self.drive_nodes = drive_nodes
        self.node_id_col = node_id_col

        # Build spatial indices
        self.walk_idx = SpatialIndex(walk_nodes, id_column=node_id_col)
        if drive_nodes is not None:
            self.drive_idx = SpatialIndex(drive_nodes, id_column=node_id_col)

    def calculate_walk(
        self,
        residential: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        max_time_min: float = MAX_WALK_TIME_MIN
    ) -> np.ndarray:
        """Calculate walking accessibility."""
        return calculate_walk_accessibility(
            residential, pois,
            self.walk_graph, self.walk_nodes,
            self.node_id_col, max_time_min
        )

    def calculate_drive(
        self,
        residential: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        max_time_min: float = MAX_WALK_TIME_MIN,
        max_snap_dist: float = MAX_DRIVE_DIST_M
    ) -> np.ndarray:
        """Calculate driving accessibility."""
        if self.drive_graph is None:
            raise ValueError("Driving network not initialized")

        return calculate_drive_accessibility(
            residential, pois,
            self.drive_graph, self.drive_nodes,
            self.node_id_col, max_time_min, max_snap_dist
        )
