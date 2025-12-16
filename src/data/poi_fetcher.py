# -*- coding: utf-8 -*-
"""
POI (Point of Interest) fetching from OpenStreetMap.
"""

import pandas as pd
import geopandas as gpd
import osmnx as ox
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import (
    PLACE_NAME,
    CRS_HK,
    CRS_WGS84,
    OSM_TAGS,
    DATA_DIR
)


def _compute_centroids_safely(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute centroids after projecting to a projected CRS to avoid warnings.

    Args:
        gdf: GeoDataFrame in WGS84

    Returns:
        GeoDataFrame with centroid geometries in Hong Kong CRS
    """
    # Project to Web Mercator for accurate centroid calculation
    gdf = gdf.to_crs("EPSG:3857")
    gdf["geometry"] = gdf.geometry.centroid
    # Project to Hong Kong CRS
    return gdf.to_crs(CRS_HK)


def fetch_osm_features(
    place: str,
    tags: Dict,
    type_name: str
) -> gpd.GeoDataFrame:
    """
    Fetch features from OSM using given tags.

    Args:
        place: Place name for query
        tags: OSM tags dictionary
        type_name: Type name to assign to features

    Returns:
        GeoDataFrame with name, geometry, and type columns
    """
    try:
        gdf = ox.features_from_place(place, tags)
        if gdf.empty:
            return gpd.GeoDataFrame(columns=["name", "geometry", "type"], crs=CRS_WGS84)

        # Keep only essential columns
        cols = [c for c in ["name", "geometry"] if c in gdf.columns]
        out = gdf[cols].copy()
        out["type"] = type_name

        return out.reset_index(drop=True)

    except Exception as e:
        print(f"Warning: Failed to fetch {type_name}: {e}")
        return gpd.GeoDataFrame(columns=["name", "geometry", "type"], crs=CRS_WGS84)


def fetch_transit_pois(
    place: str = PLACE_NAME,
    output_path: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Fetch all public transit POIs (MTR, bus, ferry, tram, minibus).

    Args:
        place: Place name for query
        output_path: Optional path to save output

    Returns:
        GeoDataFrame with all transit POIs
    """
    print("Fetching transit POIs...")

    transit_tags = OSM_TAGS["transit"]
    results = []

    for type_name, tags in transit_tags.items():
        print(f"  Fetching {type_name}...")
        gdf = fetch_osm_features(place, tags, type_name)
        results.append(gdf)
        print(f"    Found {len(gdf)} features")

    # Combine all
    combined = pd.concat(results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=CRS_WGS84)

    # Remove duplicates
    combined = combined.drop_duplicates()

    # Convert polygons to centroids (project first to avoid warning)
    combined = _compute_centroids_safely(combined)

    print(f"Total transit POIs: {len(combined)}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_file(output_path, driver="GPKG")
        print(f"Saved to {output_path}")

    return combined


def fetch_healthcare_pois(
    place: str = PLACE_NAME,
    output_path: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Fetch healthcare POIs (hospitals, clinics).

    Args:
        place: Place name for query
        output_path: Optional path to save output

    Returns:
        GeoDataFrame with healthcare POIs
    """
    print("Fetching healthcare POIs...")

    healthcare_tags = OSM_TAGS["healthcare"]
    results = []

    for type_name, tags in healthcare_tags.items():
        print(f"  Fetching {type_name}...")
        gdf = fetch_osm_features(place, tags, type_name)
        results.append(gdf)
        print(f"    Found {len(gdf)} features")

    combined = pd.concat(results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=CRS_WGS84)
    combined = combined.drop_duplicates()
    combined = _compute_centroids_safely(combined)

    print(f"Total healthcare POIs: {len(combined)}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_file(output_path, driver="GPKG")
        print(f"Saved to {output_path}")

    return combined


def fetch_daily_shopping_pois(
    place: str = PLACE_NAME,
    output_path: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Fetch daily shopping POIs (supermarkets, convenience stores, markets).

    Args:
        place: Place name for query
        output_path: Optional path to save output

    Returns:
        GeoDataFrame with daily shopping POIs
    """
    print("Fetching daily shopping POIs...")

    shopping_tags = OSM_TAGS["daily_shopping"]
    results = []

    for type_name, tags in shopping_tags.items():
        print(f"  Fetching {type_name}...")
        gdf = fetch_osm_features(place, tags, type_name)
        results.append(gdf)
        print(f"    Found {len(gdf)} features")

    combined = pd.concat(results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=CRS_WGS84)
    combined = combined.drop_duplicates()
    combined = _compute_centroids_safely(combined)

    print(f"Total daily shopping POIs: {len(combined)}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_file(output_path, driver="GPKG")
        print(f"Saved to {output_path}")

    return combined


def fetch_convenience_service_pois(
    place: str = PLACE_NAME,
    output_path: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Fetch convenience service POIs (banks, ATMs, pharmacies, post offices).

    Args:
        place: Place name for query
        output_path: Optional path to save output

    Returns:
        GeoDataFrame with convenience service POIs
    """
    print("Fetching convenience service POIs...")

    service_tags = OSM_TAGS["convenience_services"]
    results = []

    for type_name, tags in service_tags.items():
        print(f"  Fetching {type_name}...")
        gdf = fetch_osm_features(place, tags, type_name)
        results.append(gdf)
        print(f"    Found {len(gdf)} features")

    combined = pd.concat(results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=CRS_WGS84)
    combined = combined.drop_duplicates()
    combined = _compute_centroids_safely(combined)

    print(f"Total convenience service POIs: {len(combined)}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_file(output_path, driver="GPKG")
        print(f"Saved to {output_path}")

    return combined


def fetch_leisure_recreation_pois(
    place: str = PLACE_NAME,
    output_path: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Fetch leisure and recreation POIs (parks, libraries, community centers).

    Args:
        place: Place name for query
        output_path: Optional path to save output

    Returns:
        GeoDataFrame with leisure POIs
    """
    print("Fetching leisure/recreation POIs...")

    leisure_tags = OSM_TAGS["leisure_recreation"]
    results = []

    for type_name, tags in leisure_tags.items():
        print(f"  Fetching {type_name}...")
        gdf = fetch_osm_features(place, tags, type_name)
        results.append(gdf)
        print(f"    Found {len(gdf)} features")

    combined = pd.concat(results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=CRS_WGS84)
    combined = combined.drop_duplicates()
    combined = _compute_centroids_safely(combined)

    print(f"Total leisure/recreation POIs: {len(combined)}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_file(output_path, driver="GPKG")
        print(f"Saved to {output_path}")

    return combined


def fetch_all_pois(
    place: str = PLACE_NAME,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Fetch all POI categories.

    Args:
        place: Place name for query
        output_dir: Directory to save output files

    Returns:
        Dictionary mapping category names to GeoDataFrames
    """
    output_dir = Path(output_dir) if output_dir else DATA_DIR

    results = {
        "transit": fetch_transit_pois(
            place, output_dir / "HK_transit.gpkg"
        ),
        "healthcare": fetch_healthcare_pois(
            place, output_dir / "health_2326.gpkg"
        ),
        "daily_shopping": fetch_daily_shopping_pois(
            place, output_dir / "poi_daily_shopping_2326.gpkg"
        ),
        "convenience_services": fetch_convenience_service_pois(
            place, output_dir / "poi_convenience_services_2326.gpkg"
        ),
        "leisure_recreation": fetch_leisure_recreation_pois(
            place, output_dir / "poi_leisure_recreation_2326.gpkg"
        ),
    }

    return results


def fetch_buildings(
    place: str = PLACE_NAME,
    output_path: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Fetch all buildings from OSM and extract centroids.

    Args:
        place: Place name for query
        output_path: Optional path to save output

    Returns:
        GeoDataFrame with building centroids
    """
    print("Fetching buildings from OSM...")

    buildings = ox.features_from_place(place, {"building": True})
    print(f"Total buildings downloaded: {len(buildings)}")

    # Keep only polygons
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    print(f"Valid building polygons: {len(buildings)}")

    # Project to web mercator for accurate centroid
    buildings = buildings.to_crs("EPSG:3857")
    buildings["geometry"] = buildings.centroid

    # Convert to Hong Kong CRS
    buildings = buildings.to_crs(CRS_HK)

    # Keep only essential columns
    if "building" not in buildings.columns:
        buildings["building"] = None

    buildings = buildings[["building", "geometry"]].copy()

    print(f"Final building count: {len(buildings)}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        buildings.to_file(output_path, driver="GPKG", index=False)
        print(f"Saved to {output_path}")

    return buildings


def fetch_tpu_boundaries(
    output_path: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Fetch TPU (Tertiary Planning Unit) boundaries from HK government open data.

    Note: This downloads from data.gov.hk CSDI service.

    Args:
        output_path: Optional path to save output

    Returns:
        GeoDataFrame with TPU boundaries
    """
    import urllib.request
    import zipfile
    import tempfile
    import os

    print("Fetching TPU boundaries from HK government data...")

    # HK CSDI TPU data URL (Small Tertiary Planning Unit Group)
    # This is the 2021 census boundary data
    url = "https://www.censtatd.gov.hk/en/data/stat_report/gis/small_area/STPUG_boundary.zip"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "tpu.zip")

            print(f"  Downloading from {url}...")
            urllib.request.urlretrieve(url, zip_path)

            print("  Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)

            # Find the shapefile
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise FileNotFoundError("No shapefile found in downloaded archive")

            shp_path = os.path.join(tmpdir, shp_files[0])
            print(f"  Loading {shp_files[0]}...")

            tpu = gpd.read_file(shp_path)

            # Ensure correct CRS
            if tpu.crs is None:
                tpu = tpu.set_crs(CRS_HK)
            else:
                tpu = tpu.to_crs(CRS_HK)

            print(f"Loaded {len(tpu)} TPU boundaries")

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as shapefile (multiple files)
                if str(output_path).endswith('.shp'):
                    tpu.to_file(output_path)
                else:
                    tpu.to_file(output_path, driver="GPKG")
                print(f"Saved to {output_path}")

            return tpu

    except Exception as e:
        print(f"\nError fetching TPU boundaries: {e}")
        print("\nPlease manually download the TPU boundary shapefile from:")
        print("  https://data.gov.hk/en-data/dataset/hk-censtatd-censtatd_gis-sma-boundary")
        print("\nAnd place these files in the data/ directory:")
        print("  - STPUG_21C_converted.shp")
        print("  - STPUG_21C_converted.shx")
        print("  - STPUG_21C_converted.dbf")
        print("  - STPUG_21C_converted.prj")
        raise
