# -*- coding: utf-8 -*-
"""
Driving network construction and management.
"""

import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
from pathlib import Path
from typing import Optional, Tuple, Union

from ..config import (
    CRS_HK,
    DEFAULT_DRIVE_SPEED_MPS,
    PLACE_NAME
)


def build_drive_network(
    place: str = PLACE_NAME,
    drive_speed_mps: float = DEFAULT_DRIVE_SPEED_MPS,
    output_path: Optional[Union[str, Path]] = None
) -> nx.MultiDiGraph:
    """
    Build driving network from OpenStreetMap.

    Args:
        place: Place name for OSM query
        drive_speed_mps: Driving speed in meters per second
        output_path: Path to save GraphML (optional)

    Returns:
        NetworkX MultiDiGraph projected to Hong Kong CRS
    """
    print(f"Downloading driving network for {place}...")

    # Download driving network
    G = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True
    )

    # Project to Hong Kong CRS
    print(f"Projecting to {CRS_HK}...")
    G = ox.project_graph(G, to_crs=CRS_HK)

    # Recompute edge lengths in projected CRS
    G = ox.distance.add_edge_lengths(G)

    # Compute travel time
    for u, v, k, d in G.edges(keys=True, data=True):
        d["travel_time"] = d["length"] / drive_speed_mps

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ox.io.save_graphml(G, output_path)
        print(f"Saved to {output_path}")

    return G


def load_drive_network(
    graphml_path: Union[str, Path]
) -> nx.MultiDiGraph:
    """
    Load driving network from GraphML file.

    Args:
        graphml_path: Path to GraphML file

    Returns:
        NetworkX MultiDiGraph
    """
    print(f"Loading driving network from {graphml_path}...")
    G = ox.io.load_graphml(graphml_path)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_drive_network_nodes(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Extract nodes GeoDataFrame from driving network graph.

    Args:
        G: NetworkX graph

    Returns:
        GeoDataFrame of nodes
    """
    nodes, _ = ox.graph_to_gdfs(G, nodes=True, edges=False)
    return nodes.reset_index()
