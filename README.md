# ELCI - Elderly Living Convenience Index Analysis

A modular Python package for computing the Elderly Living Convenience Index (ELCI) for Hong Kong's Tertiary Planning Units (TPUs).

## Project Structure

```
elder/
├── main.py                 # Original Colab script (preserved)
├── run_pipeline.py         # Main pipeline entry point
├── requirements.txt        # Python dependencies
├── README_PROJECT.md       # This file
│
├── src/                    # Source code package
│   ├── __init__.py
│   ├── config.py           # Configuration and constants
│   │
│   ├── data/               # Data loading and POI fetching
│   │   ├── __init__.py
│   │   ├── poi_fetcher.py  # Fetch POIs from OpenStreetMap
│   │   └── loaders.py      # Load various data files
│   │
│   ├── network/            # Network construction
│   │   ├── __init__.py
│   │   ├── walk_network.py # Walking network building
│   │   ├── drive_network.py# Driving network building
│   │   └── graph_builder.py# NetworkX graph utilities
│   │
│   ├── accessibility/      # Accessibility calculation
│   │   ├── __init__.py
│   │   ├── calculator.py   # Core accessibility algorithms
│   │   └── aggregator.py   # TPU-level aggregation
│   │
│   ├── analysis/           # Analysis modules
│   │   ├── __init__.py
│   │   ├── elci.py         # ELCI calculation
│   │   ├── clustering.py   # K-means and hierarchical clustering
│   │   ├── mismatch.py     # Priority zone identification
│   │   └── regression.py   # Population prediction models
│   │
│   ├── visualization/      # Visualization
│   │   ├── __init__.py
│   │   ├── maps.py         # Choropleth and spatial maps
│   │   └── charts.py       # Heatmaps, scatter plots, etc.
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── data_utils.py   # Data processing utilities
│       └── geo_utils.py    # Geospatial utilities
│
├── data/                   # Input data directory
│   ├── STPUG_21C_converted.shp    # TPU boundaries
│   ├── residential_buildings_2326.gpkg
│   ├── hk_walk_nodes_2326.gpkg
│   ├── hk_walk_edges_2326.gpkg
│   └── ...
│
└── output/                 # Output directory
    ├── figures/            # Generated visualizations
    └── *.xlsx              # Analysis results
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

```bash
python run_pipeline.py --full
```

### Running Individual Steps

```bash
# Fetch POIs from OpenStreetMap
python run_pipeline.py --fetch-pois

# Only compute ELCI from existing accessibility data
python run_pipeline.py --elci-only

# Only run clustering analysis
python run_pipeline.py --cluster-only

# Only generate visualizations
python run_pipeline.py --viz-only
```

### Using as a Library

```python
from src.data.loaders import load_tpu_boundaries, load_residential_buildings
from src.accessibility.calculator import AccessibilityCalculator
from src.analysis.elci import ELCICalculator

# Load data
tpu = load_tpu_boundaries()
residential = load_residential_buildings()

# Calculate ELCI
calculator = ELCICalculator()
result = calculator.calculate(accessibility_df)
```

## ELCI Methodology

The ELCI (Elderly Living Convenience Index) is computed from six accessibility dimensions:

1. **Transit Walk** - Walking time to nearest public transit (MTR, bus, ferry, tram)
2. **Hospital Walk** - Walking time to nearest healthcare facility
3. **Hospital Drive** - Driving time to nearest healthcare facility
4. **Daily Walk** - Walking time to nearest daily shopping (supermarket, market)
5. **Service Walk** - Walking time to nearest convenience service (bank, pharmacy)
6. **Leisure Walk** - Walking time to nearest leisure facility (park, library)

### Calculation Steps:

1. **Network Analysis**: Multi-source Dijkstra algorithm computes shortest paths
2. **Building-level**: Each residential building gets accessibility times
3. **TPU Aggregation**: Building-level times are averaged per TPU
4. **Normalization**: Each dimension is min-max normalized (reversed, as lower time = better)
5. **ELCI Score**: Equal-weighted average scaled to 0-100

## Configuration

Key parameters can be modified in `src/config.py`:

- `ELDERLY_WALK_SPEED_MPS`: Walking speed (default: 0.8 m/s)
- `DEFAULT_DRIVE_SPEED_KMH`: Urban driving speed (default: 20 km/h)
- `MAX_WALK_TIME_MIN`: Maximum walking time threshold (default: 60 min)
- `DEFAULT_N_CLUSTERS`: Number of clusters for K-means (default: 4)
- `ELCI_WEIGHTS`: Dimension weights for ELCI calculation

## Data Requirements

### Required Input Files:

1. **TPU Boundaries**: `STPUG_21C_converted.shp` - Hong Kong TPU polygons
2. **Residential Buildings**: Building centroids in EPSG:2326
3. **Walking Network**: Nodes and edges from HK CSDI PedestrianRoute or OSM

### Optional:
- Population projection data for mismatch analysis

## Output Files

- `TPU_accessibility_all.xlsx` - All accessibility metrics by TPU
- `TPU_ELCI_result.xlsx` - ELCI scores with normalized dimensions
- `TPU_cluster_result.xlsx` - Clustering results
- `mismatch_analysis.xlsx` - Priority zone identification
- `output/figures/*.png` - Generated maps and charts

## Key Classes

### AccessibilityCalculator
```python
calculator = AccessibilityCalculator(walk_graph, walk_nodes, drive_graph, drive_nodes)
times = calculator.calculate_walk(residential, pois)
```

### ELCICalculator
```python
elci = ELCICalculator(dimension_cols, weights)
result = elci.calculate(accessibility_df)
```

### ClusterAnalyzer
```python
analyzer = ClusterAnalyzer(n_clusters=4)
result = analyzer.fit_kmeans(elci_df)
profiles = analyzer.get_profiles()
```

### MismatchAnalyzer
```python
analyzer = MismatchAnalyzer()
result = analyzer.analyze(elci_df, population_df, population_col="POP65PLUS_2026_pred")
ranking = analyzer.get_priority_ranking(top_n=20)
```

## References

Based on the research methodology described in ClAUDE.md for analyzing elderly living convenience in Hong Kong TPUs.
