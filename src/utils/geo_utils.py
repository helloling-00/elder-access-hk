# -*- coding: utf-8 -*-
"""
Geospatial utility functions.
"""

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
from shapely.geometry import Point, LineString, MultiLineString
from typing import Tuple, Optional, List, Dict, Any


class SpatialIndex:
    """
    KD-Tree based spatial index for fast nearest neighbor queries.
    """

    def __init__(self, gdf: gpd.GeoDataFrame, id_column: Optional[str] = None):
        """
        Initialize spatial index from GeoDataFrame.

        Args:
            gdf: GeoDataFrame with point geometries
            id_column: Column to use as node IDs (uses index if None)
        """
        self.gdf = gdf

        # Extract coordinates
        xy = np.column_stack([gdf.geometry.x, gdf.geometry.y])
        self.tree = cKDTree(xy)

        # Store IDs
        if id_column and id_column in gdf.columns:
            self.ids = gdf[id_column].values
        else:
            self.ids = gdf.index.to_numpy()

    def nearest(self, x: float, y: float, max_dist: Optional[float] = None) -> Optional[Any]:
        """
        Find nearest node ID to given coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            max_dist: Maximum distance threshold (returns None if exceeded)

        Returns:
            Node ID or None if no node within max_dist
        """
        dist, idx = self.tree.query([x, y])

        if max_dist is not None and dist > max_dist:
            return None

        return self.ids[idx]

    def nearest_from_geometry(
        self,
        geom: Point,
        max_dist: Optional[float] = None
    ) -> Optional[Any]:
        """
        Find nearest node ID to given geometry.

        Args:
            geom: Point geometry
            max_dist: Maximum distance threshold

        Returns:
            Node ID or None
        """
        return self.nearest(geom.x, geom.y, max_dist)

    def snap_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        max_dist: Optional[float] = None,
        geometry_column: str = "geometry"
    ) -> List[Optional[Any]]:
        """
        Snap all points in a GeoDataFrame to nearest nodes.

        Args:
            gdf: GeoDataFrame with point geometries
            max_dist: Maximum distance threshold
            geometry_column: Column containing geometries

        Returns:
            List of node IDs (None for points beyond max_dist)
        """
        geoms = gdf[geometry_column]
        return [self.nearest_from_geometry(g, max_dist) for g in geoms]


def assign_points_to_polygons(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    polygon_id_col: str,
    point_geom_col: str = "geometry"
) -> List[Optional[Any]]:
    """
    Assign points to polygons using STRtree spatial index.

    Args:
        points_gdf: GeoDataFrame with point geometries
        polygons_gdf: GeoDataFrame with polygon geometries
        polygon_id_col: Column in polygons_gdf containing polygon IDs
        point_geom_col: Column in points_gdf containing geometries

    Returns:
        List of polygon IDs (None for points not in any polygon)
    """
    # Build spatial index
    poly_geoms = np.array(polygons_gdf.geometry.values)
    poly_ids = np.array(polygons_gdf[polygon_id_col].values)

    tree = STRtree(poly_geoms)
    geom_to_id = {id(g): pid for g, pid in zip(poly_geoms, poly_ids)}

    # Assign points
    assigned = []
    for pt in points_gdf[point_geom_col]:
        hits = tree.query(pt)
        found_id = None

        for idx in hits:
            poly = poly_geoms[idx]
            if poly.contains(pt) or pt.within(poly):
                found_id = geom_to_id[id(poly)]
                break

        assigned.append(found_id)

    return assigned


def drop_z_dimension(geom) -> Optional[LineString]:
    """
    Remove Z dimension from LineString or MultiLineString.

    Args:
        geom: Input geometry (LineString or MultiLineString with Z)

    Returns:
        2D geometry or None if unsupported type
    """
    if geom.geom_type == "LineString":
        return LineString([(x, y) for x, y, *_ in geom.coords])
    elif geom.geom_type == "MultiLineString":
        return MultiLineString([
            LineString([(x, y) for x, y, *_ in line.coords])
            for line in geom.geoms
        ])
    return None


def compute_geometry_centroid(gdf: gpd.GeoDataFrame, inplace: bool = True) -> gpd.GeoDataFrame:
    """
    Replace polygon/multipolygon geometries with their centroids.

    Args:
        gdf: Input GeoDataFrame
        inplace: Modify in place

    Returns:
        GeoDataFrame with centroid geometries
    """
    if not inplace:
        gdf = gdf.copy()

    gdf["geometry"] = gdf.geometry.centroid
    return gdf


def filter_by_geometry_type(
    gdf: gpd.GeoDataFrame,
    geom_types: List[str]
) -> gpd.GeoDataFrame:
    """
    Filter GeoDataFrame by geometry type.

    Args:
        gdf: Input GeoDataFrame
        geom_types: List of allowed geometry types (e.g., ["Point", "MultiPoint"])

    Returns:
        Filtered GeoDataFrame
    """
    return gdf[gdf.geometry.type.isin(geom_types)]


def extract_nodes_and_edges(
    gdf: gpd.GeoDataFrame,
    walk_speed_mps: float = 0.8
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Extract nodes and edges from a GeoDataFrame of LineStrings.

    Args:
        gdf: GeoDataFrame with LineString geometries
        walk_speed_mps: Walking speed in meters per second

    Returns:
        Tuple of (nodes_gdf, edges_gdf)
    """
    # Collect all coordinates
    coords_list = []
    for geom in gdf.geometry:
        coords_list.append(np.asarray(geom.coords))

    all_coords = np.vstack(coords_list)
    unique_coords, inverse = np.unique(all_coords, axis=0, return_inverse=True)

    # Create nodes
    nodes_gdf = gpd.GeoDataFrame(
        {"osmid": np.arange(len(unique_coords))},
        geometry=[Point(x, y) for x, y in unique_coords],
        crs=gdf.crs
    )

    # Create edges
    rows = []
    offset = 0

    for geom in gdf.geometry:
        coords = np.asarray(geom.coords)
        n = len(coords)
        idxs = inverse[offset: offset + n]

        for a, b in zip(idxs[:-1], idxs[1:]):
            line = LineString([unique_coords[a], unique_coords[b]])
            rows.append([a, b, line])

        offset += n

    edges_gdf = gpd.GeoDataFrame(
        rows,
        columns=["u", "v", "geometry"],
        crs=gdf.crs
    )

    # Compute length and travel time
    edges_gdf["length"] = edges_gdf.length
    edges_gdf["travel_time"] = edges_gdf["length"] / walk_speed_mps

    return nodes_gdf, edges_gdf
