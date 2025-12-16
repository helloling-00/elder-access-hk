# -*- coding: utf-8 -*-
"""
NetworkX graph building utilities.
"""

import networkx as nx
import geopandas as gpd
from typing import Optional
from tqdm import tqdm


def build_networkx_graph(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    node_id_col: str = "osmid",
    show_progress: bool = True
) -> nx.MultiDiGraph:
    """
    Build NetworkX MultiDiGraph from nodes and edges GeoDataFrames.

    Args:
        nodes: GeoDataFrame with node information
        edges: GeoDataFrame with edge information (must have 'u', 'v' columns)
        node_id_col: Column name for node IDs
        show_progress: Whether to show progress bars

    Returns:
        NetworkX MultiDiGraph
    """
    G = nx.MultiDiGraph()

    # Add nodes
    node_iter = nodes.itertuples()
    if show_progress:
        node_iter = tqdm(node_iter, total=len(nodes), desc="Adding nodes")

    for row in node_iter:
        node_id = getattr(row, node_id_col)
        G.add_node(node_id, x=row.geometry.x, y=row.geometry.y)

    # Add edges
    edge_iter = edges.itertuples()
    if show_progress:
        edge_iter = tqdm(edge_iter, total=len(edges), desc="Adding edges")

    for row in edge_iter:
        edge_attrs = {
            "length": getattr(row, "length", None),
            "travel_time": getattr(row, "travel_time", None),
            "geometry": row.geometry
        }
        # Remove None values
        edge_attrs = {k: v for k, v in edge_attrs.items() if v is not None}
        G.add_edge(row.u, row.v, **edge_attrs)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def get_largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Get the largest weakly connected component of a graph.

    Args:
        G: Input graph

    Returns:
        Subgraph containing only the largest component
    """
    if nx.is_weakly_connected(G):
        return G

    components = list(nx.weakly_connected_components(G))
    largest = max(components, key=len)

    print(f"Extracting largest component: {len(largest)} of {G.number_of_nodes()} nodes")

    return G.subgraph(largest).copy()


def compute_travel_times(
    G: nx.MultiDiGraph,
    speed_mps: float,
    length_attr: str = "length",
    time_attr: str = "travel_time"
) -> nx.MultiDiGraph:
    """
    Compute travel times for all edges based on length and speed.

    Args:
        G: Input graph
        speed_mps: Speed in meters per second
        length_attr: Edge attribute containing length
        time_attr: Edge attribute to store travel time

    Returns:
        Graph with updated travel times
    """
    for u, v, k, d in G.edges(keys=True, data=True):
        if length_attr in d:
            d[time_attr] = d[length_attr] / speed_mps

    return G
