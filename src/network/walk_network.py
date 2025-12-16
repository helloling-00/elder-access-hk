# -*- coding: utf-8 -*-
"""
Walking network construction and management.
"""

import os
import numpy as np
import geopandas as gpd
import networkx as nx
from pathlib import Path
from typing import Optional, Tuple, Union
from tqdm import tqdm

from ..config import (
    CRS_HK,
    ELDERLY_WALK_SPEED_MPS,
    DATA_DIR,
    FILE_PATHS
)
from ..utils.geo_utils import drop_z_dimension, extract_nodes_and_edges


def load_pedestrian_route_from_gdb(
    gdb_path: Union[str, Path],
    layer: str = "PedestrianRoute"
) -> gpd.GeoDataFrame:
    """
    Load pedestrian route layer from geodatabase.

    Args:
        gdb_path: Path to .gdb directory
        layer: Layer name to load

    Returns:
        GeoDataFrame with pedestrian routes
    """
    from pyogrio import read_dataframe

    print(f"Loading {layer} from {gdb_path}...")
    gdf = read_dataframe(gdb_path, layer=layer)
    print(f"Loaded {len(gdf)} features")

    return gdf


def build_walk_network_from_gdb(
    gdb_path: Union[str, Path],
    walk_speed_mps: float = ELDERLY_WALK_SPEED_MPS,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build walking network from HK CSDI PedestrianRoute geodatabase.

    Args:
        gdb_path: Path to geodatabase containing PedestrianRoute layer
        walk_speed_mps: Walking speed in meters per second
        output_dir: Directory to save output files (optional)

    Returns:
        Tuple of (nodes_gdf, edges_gdf)
    """
    # Load data
    gdf = load_pedestrian_route_from_gdb(gdb_path)

    # Drop Z dimension
    print("Dropping Z dimension...")
    gdf["geom2"] = gdf.geometry.apply(drop_z_dimension)
    gdf = gdf[gdf["geom2"].notna()]
    gdf = gdf.set_geometry("geom2")

    # Explode MultiLineStrings
    print("Exploding MultiLineStrings...")
    gdf = gdf.explode(index_parts=False)
    gdf = gdf.set_geometry("geom2")
    print(f"LineStrings: {len(gdf)}")

    # Extract nodes and edges
    print("Extracting nodes and edges...")
    nodes_gdf, edges_gdf = extract_nodes_and_edges(gdf, walk_speed_mps)

    print(f"Nodes: {len(nodes_gdf)}")
    print(f"Edges: {len(edges_gdf)}")

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nodes_path = output_dir / FILE_PATHS["walk_nodes"]
        edges_path = output_dir / FILE_PATHS["walk_edges"]

        print(f"Saving nodes to {nodes_path}...")
        nodes_gdf.to_file(nodes_path, driver="GPKG")

        print(f"Saving edges to {edges_path}...")
        edges_gdf.to_file(edges_path, driver="GPKG")

    return nodes_gdf, edges_gdf


def build_walk_network_from_osm(
    place: str = "Hong Kong, China",
    walk_speed_mps: float = ELDERLY_WALK_SPEED_MPS,
    output_path: Optional[Union[str, Path]] = None
) -> nx.MultiDiGraph:
    """
    Build walking network from OpenStreetMap.

    Args:
        place: Place name for OSM query
        walk_speed_mps: Walking speed in meters per second
        output_path: Path to save GraphML (optional)

    Returns:
        NetworkX MultiDiGraph
    """
    import osmnx as ox

    print(f"Downloading walking network for {place}...")

    # Download and project
    G = ox.graph_from_place(place, network_type="walk", simplify=True)
    G = ox.project_graph(G, to_crs=CRS_HK)

    # Add edge lengths
    G = ox.distance.add_edge_lengths(G)

    # Compute travel time
    for u, v, k, d in G.edges(keys=True, data=True):
        d["travel_time"] = d["length"] / walk_speed_mps

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ox.io.save_graphml(G, output_path)
        print(f"Saved to {output_path}")

    return G


def load_walk_network(
    nodes_path: Optional[Union[str, Path]] = None,
    edges_path: Optional[Union[str, Path]] = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load walking network from saved files.

    Args:
        nodes_path: Path to nodes GPKG
        edges_path: Path to edges GPKG

    Returns:
        Tuple of (nodes_gdf, edges_gdf)
    """
    nodes_path = nodes_path or DATA_DIR / FILE_PATHS["walk_nodes"]
    edges_path = edges_path or DATA_DIR / FILE_PATHS["walk_edges"]

    print(f"Loading nodes from {nodes_path}...")
    nodes = gpd.read_file(nodes_path)

    print(f"Loading edges from {edges_path}...")
    edges = gpd.read_file(edges_path)

    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")

    return nodes, edges
