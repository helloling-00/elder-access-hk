# -*- coding: utf-8 -*-
"""
Configuration module for ELCI analysis.
Contains all constants, paths, and parameter settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"
CRS_HK = "EPSG:2326"  # Hong Kong 1980 Grid System
CRS_WEB_MERCATOR = "EPSG:3857"

# Study area
PLACE_NAME = "Hong Kong, China"

# Walking parameters
ELDERLY_WALK_SPEED_MPS = 0.8  # meters per second (elderly walking speed)
DEFAULT_WALK_SPEED_MPS = 1.2  # meters per second (normal walking speed)

# Driving parameters
DEFAULT_DRIVE_SPEED_KMH = 20  # km/h for urban driving
DEFAULT_DRIVE_SPEED_MPS = DEFAULT_DRIVE_SPEED_KMH * 1000 / 3600

# Accessibility thresholds
MAX_WALK_TIME_MIN = 60  # maximum walking time in minutes
MAX_DRIVE_DIST_M = 300  # maximum distance for snapping to drive network

# ELCI dimension weights (equal weights by default)
ELCI_WEIGHTS = {
    "transit_walk": 1.0,
    "hospital_walk": 1.0,
    "hospital_drive": 1.0,
    "daily_walk": 1.0,
    "service_walk": 1.0,
    "leisure_walk": 1.0,
}

# Clustering parameters
DEFAULT_N_CLUSTERS = 4

# OSM tags for different facility types
OSM_TAGS = {
    # Public transportation
    "transit": {
        "mtr": {"railway": "station", "subway": "yes"},
        "ferry": {"amenity": "ferry_terminal"},
        "bus": {"highway": "bus_stop"},
        "minibus": {"public_transport": "platform", "minibus": "yes"},
        "tram": {"railway": "tram_stop"},
    },
    # Healthcare facilities
    "healthcare": {
        "hospital": {"amenity": "hospital"},
        "clinic_amenity": {"amenity": "clinic"},
        "clinic_healthcare": {"healthcare": "clinic"},
    },
    # Daily shopping
    "daily_shopping": {
        "supermarket": {"shop": "supermarket"},
        "convenience": {"shop": "convenience"},
        "market": {"amenity": "marketplace"},
        "mall": {"shop": "mall"},
    },
    # Convenience services
    "convenience_services": {
        "bank": {"amenity": "bank"},
        "atm": {"amenity": "atm"},
        "pharmacy": {"amenity": "pharmacy"},
        "post_office": {"amenity": "post_office"},
    },
    # Leisure and recreation
    "leisure_recreation": {
        "park": {"leisure": "park"},
        "library": {"amenity": "library"},
        "community_centre": {"amenity": "community_centre"},
        "playground": {"leisure": "playground"},
    },
}

# File paths (default names)
FILE_PATHS = {
    "walk_nodes": "hk_walk_nodes_2326.gpkg",
    "walk_edges": "hk_walk_edges_2326.gpkg",
    "tpu_boundary": "STPUG_21C_converted.shp",
    "residential": "residential_buildings_2326.gpkg",
    "transit": "HK_transit_all.gpkg",
    "healthcare": "health_2326.gpkg",
    "daily_shopping": "poi_daily_shopping_2326.gpkg",
    "convenience_services": "poi_convenience_services_2326.gpkg",
    "leisure_recreation": "poi_leisure_recreation_2326.gpkg",
}

# Output file names
OUTPUT_FILES = {
    "transit_accessibility": "TPU_transit_walk.xlsx",
    "health_walk": "TPU_health_walk.xlsx",
    "health_drive": "TPU_health_drive.xlsx",
    "daily_walk": "TPU_daily_walk.xlsx",
    "service_walk": "TPU_service_walk.xlsx",
    "leisure_walk": "TPU_leisure_walk.xlsx",
    "elci_result": "TPU_ELCI_result.xlsx",
    "cluster_result": "TPU_cluster_result.xlsx",
    "accessibility_all": "TPU_accessibility_all.xlsx",
}
