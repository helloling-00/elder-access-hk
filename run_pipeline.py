#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELCI Analysis Pipeline
======================

Main pipeline script for computing Elderly Living Convenience Index (ELCI)
for Hong Kong TPUs (Tertiary Planning Units).

This script orchestrates the complete analysis workflow:
1. Load/build network data
2. Fetch/load POI data
3. Calculate accessibility for each dimension
4. Aggregate to TPU level
5. Compute ELCI scores
6. Perform clustering analysis
7. Identify priority zones
8. Generate visualizations

Usage:
    python run_pipeline.py --full        # Run full pipeline
    python run_pipeline.py --elci-only   # Only compute ELCI from existing accessibility data
    python run_pipeline.py --viz-only    # Only generate visualizations

Author: ELCI Research Team
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_DIR, OUTPUT_DIR, FIGURES_DIR,
    FILE_PATHS, OUTPUT_FILES
)


def run_data_preparation(fetch_tpu: bool = False):
    """Fetch and prepare all input data."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)

    from src.data.poi_fetcher import fetch_all_pois, fetch_buildings, fetch_tpu_boundaries

    # Fetch TPU boundaries if requested
    if fetch_tpu:
        print("\nFetching TPU boundaries...")
        try:
            tpu = fetch_tpu_boundaries(output_path=DATA_DIR / "STPUG_21C.gpkg")
        except Exception as e:
            print(f"Warning: Could not fetch TPU boundaries: {e}")

    # Fetch POIs
    print("\nFetching POIs from OpenStreetMap...")
    pois = fetch_all_pois(output_dir=DATA_DIR)

    # Fetch buildings
    print("\nFetching building footprints...")
    buildings = fetch_buildings(output_path=DATA_DIR / "residential_buildings_2326.gpkg")

    return pois, buildings


def run_network_building():
    """Build or load walking and driving networks."""
    print("\n" + "="*60)
    print("STEP 2: NETWORK BUILDING")
    print("="*60)

    from src.network.walk_network import load_walk_network, build_walk_network_from_osm
    from src.network.drive_network import build_drive_network
    from src.network.graph_builder import build_networkx_graph

    # Check if walk network exists
    walk_nodes_path = DATA_DIR / FILE_PATHS["walk_nodes"]
    walk_edges_path = DATA_DIR / FILE_PATHS["walk_edges"]

    if walk_nodes_path.exists() and walk_edges_path.exists():
        print("\nLoading existing walk network...")
        walk_nodes, walk_edges = load_walk_network()
        walk_graph = build_networkx_graph(walk_nodes, walk_edges)
    else:
        print("\nBuilding walk network from OSM...")
        walk_graph = build_walk_network_from_osm(
            output_path=DATA_DIR / "hk_walk_osm.graphml"
        )
        import osmnx as ox
        walk_nodes, walk_edges = ox.graph_to_gdfs(walk_graph)
        walk_nodes = walk_nodes.reset_index()

    # Build drive network
    print("\nBuilding drive network...")
    drive_graph = build_drive_network()
    import osmnx as ox
    drive_nodes, _ = ox.graph_to_gdfs(drive_graph)
    drive_nodes = drive_nodes.reset_index()

    return {
        "walk_graph": walk_graph,
        "walk_nodes": walk_nodes,
        "drive_graph": drive_graph,
        "drive_nodes": drive_nodes
    }


def run_accessibility_calculation(networks):
    """Calculate accessibility for all dimensions."""
    print("\n" + "="*60)
    print("STEP 3: ACCESSIBILITY CALCULATION")
    print("="*60)

    from src.data.loaders import (
        load_tpu_boundaries, load_residential_buildings,
        load_transit_pois, load_healthcare_pois,
        load_daily_shopping_pois, load_convenience_service_pois,
        load_leisure_recreation_pois
    )
    from src.accessibility.calculator import AccessibilityCalculator
    from src.accessibility.aggregator import TPUAggregator

    # Load data
    print("\nLoading input data...")
    tpu = load_tpu_boundaries()
    residential = load_residential_buildings()
    transit = load_transit_pois()
    healthcare = load_healthcare_pois()
    daily = load_daily_shopping_pois()
    service = load_convenience_service_pois()
    leisure = load_leisure_recreation_pois()

    # Initialize calculator
    calculator = AccessibilityCalculator(
        walk_graph=networks["walk_graph"],
        walk_nodes=networks["walk_nodes"],
        drive_graph=networks["drive_graph"],
        drive_nodes=networks["drive_nodes"]
    )

    # Calculate accessibility for each dimension
    print("\nCalculating transit accessibility (walk)...")
    residential["transit_walk_min"] = calculator.calculate_walk(residential, transit)

    print("\nCalculating healthcare accessibility (walk)...")
    residential["hospital_walk_min"] = calculator.calculate_walk(residential, healthcare)

    print("\nCalculating healthcare accessibility (drive)...")
    residential["hospital_drive_min"] = calculator.calculate_drive(residential, healthcare)

    print("\nCalculating daily shopping accessibility (walk)...")
    residential["daily_walk_min"] = calculator.calculate_walk(residential, daily)

    print("\nCalculating convenience services accessibility (walk)...")
    residential["service_walk_min"] = calculator.calculate_walk(residential, service)

    print("\nCalculating leisure accessibility (walk)...")
    residential["leisure_walk_min"] = calculator.calculate_walk(residential, leisure)

    # Aggregate to TPU level
    print("\nAggregating to TPU level...")
    aggregator = TPUAggregator(tpu)

    for col in ["transit_walk_min", "hospital_walk_min", "hospital_drive_min",
                "daily_walk_min", "service_walk_min", "leisure_walk_min"]:
        aggregator.add_accessibility(residential, col, col)

    # Save combined results
    result = aggregator.get_combined_results()
    output_path = OUTPUT_DIR / OUTPUT_FILES["accessibility_all"]
    result.to_excel(output_path, index=False)
    print(f"\nSaved accessibility results to {output_path}")

    return result, tpu


def run_elci_calculation(accessibility_df=None):
    """Calculate ELCI scores."""
    print("\n" + "="*60)
    print("STEP 4: ELCI CALCULATION")
    print("="*60)

    import pandas as pd
    from src.analysis.elci import ELCICalculator

    # Load accessibility data if not provided
    if accessibility_df is None:
        accessibility_path = OUTPUT_DIR / OUTPUT_FILES["accessibility_all"]
        if not accessibility_path.exists():
            raise FileNotFoundError(
                f"Accessibility data not found at {accessibility_path}. "
                "Run accessibility calculation first."
            )
        accessibility_df = pd.read_excel(accessibility_path)

    # Calculate ELCI
    calculator = ELCICalculator()
    result = calculator.calculate(accessibility_df)
    result = calculator.categorize()

    # Save results
    output_path = OUTPUT_DIR / OUTPUT_FILES["elci_result"]
    calculator.save(output_path)

    return result


def run_clustering_analysis(elci_df=None):
    """Perform clustering analysis."""
    print("\n" + "="*60)
    print("STEP 5: CLUSTERING ANALYSIS")
    print("="*60)

    import pandas as pd
    from src.analysis.clustering import ClusterAnalyzer

    # Load ELCI data if not provided
    if elci_df is None:
        elci_path = OUTPUT_DIR / OUTPUT_FILES["elci_result"]
        if not elci_path.exists():
            raise FileNotFoundError(
                f"ELCI data not found at {elci_path}. "
                "Run ELCI calculation first."
            )
        elci_df = pd.read_excel(elci_path)

    # Perform clustering
    analyzer = ClusterAnalyzer(n_clusters=4)

    print("\nFinding optimal k...")
    inertias = analyzer.find_optimal_k(elci_df)

    print("\nPerforming K-means clustering...")
    result = analyzer.fit_kmeans(elci_df)

    print("\nComputing cluster profiles...")
    profiles = analyzer.get_profiles()

    # Save results
    output_path = OUTPUT_DIR / OUTPUT_FILES["cluster_result"]
    result.to_excel(output_path, index=False)
    print(f"\nSaved clustering results to {output_path}")

    return result, profiles, inertias


def run_mismatch_analysis(elci_df=None, population_df=None):
    """Identify priority intervention zones."""
    print("\n" + "="*60)
    print("STEP 6: MISMATCH ANALYSIS")
    print("="*60)

    import pandas as pd
    from src.analysis.mismatch import MismatchAnalyzer

    # Load data if not provided
    if elci_df is None:
        elci_path = OUTPUT_DIR / OUTPUT_FILES["elci_result"]
        elci_df = pd.read_excel(elci_path)

    if population_df is None:
        # Look for population data
        pop_path = DATA_DIR / "tpu_65plus_pred_2026.xlsx"
        if pop_path.exists():
            population_df = pd.read_excel(pop_path)
        else:
            print("Warning: Population data not found. Skipping mismatch analysis.")
            return None

    # Perform analysis
    analyzer = MismatchAnalyzer()
    result = analyzer.analyze(
        elci_df, population_df,
        population_col="POP65PLUS_2026_pred",
        merge_on="stpug_eng"
    )

    # Get priority ranking
    ranking = analyzer.get_priority_ranking(top_n=20)

    # Save results
    output_path = OUTPUT_DIR / "mismatch_analysis.xlsx"
    analyzer.save(output_path)

    return result, ranking


def run_visualizations(elci_df=None, cluster_df=None, tpu_gdf=None):
    """Generate all visualizations."""
    print("\n" + "="*60)
    print("STEP 7: VISUALIZATION")
    print("="*60)

    import pandas as pd
    from src.data.loaders import load_tpu_boundaries
    from src.visualization.maps import (
        plot_elci_map, plot_cluster_map, plot_priority_zones,
        plot_side_by_side_maps
    )
    from src.visualization.charts import (
        plot_cluster_heatmap, plot_scatter_population_elci,
        plot_elbow_curve
    )

    # Load data if not provided
    if tpu_gdf is None:
        tpu_gdf = load_tpu_boundaries()

    if elci_df is None:
        elci_path = OUTPUT_DIR / OUTPUT_FILES["elci_result"]
        if elci_path.exists():
            elci_df = pd.read_excel(elci_path)

    if cluster_df is None:
        cluster_path = OUTPUT_DIR / OUTPUT_FILES["cluster_result"]
        if cluster_path.exists():
            cluster_df = pd.read_excel(cluster_path)

    # Merge data with TPU geometries
    if elci_df is not None:
        tpu_merged = tpu_gdf.merge(elci_df, on="stpug_eng", how="left")

        # ELCI map
        print("\nGenerating ELCI map...")
        plot_elci_map(tpu_merged, output_path=FIGURES_DIR / "ELCI_map.png")

    if cluster_df is not None:
        tpu_cluster = tpu_gdf.merge(cluster_df, on="stpug_eng", how="left")

        # Cluster map
        print("\nGenerating cluster map...")
        plot_cluster_map(tpu_cluster, output_path=FIGURES_DIR / "cluster_map.png")

        # Cluster heatmap
        from src.analysis.clustering import compute_cluster_profiles, DEFAULT_CLUSTER_FEATURES
        valid_cols = [c for c in DEFAULT_CLUSTER_FEATURES if c in cluster_df.columns]
        if valid_cols:
            profiles = cluster_df.groupby("cluster_kmeans")[valid_cols].mean()
            plot_cluster_heatmap(
                profiles,
                pretty_names={
                    "transit_walk_min_norm": "Transit",
                    "hospital_walk_min_norm": "Hosp-Walk",
                    "hospital_drive_min_norm": "Hosp-Drive",
                    "daily_walk_min_norm": "Daily",
                    "service_walk_min_norm": "Service",
                    "leisure_walk_min_norm": "Leisure"
                },
                output_path=FIGURES_DIR / "cluster_heatmap.png"
            )

    # Priority zones (if mismatch analysis was run)
    mismatch_path = OUTPUT_DIR / "mismatch_analysis.xlsx"
    if mismatch_path.exists():
        mismatch_df = pd.read_excel(mismatch_path)
        tpu_priority = tpu_gdf.merge(mismatch_df, on="stpug_eng", how="left")

        if "priority_zone" in tpu_priority.columns:
            print("\nGenerating priority zones map...")
            plot_priority_zones(
                tpu_priority,
                output_path=FIGURES_DIR / "priority_zones.png"
            )

    print(f"\nAll figures saved to {FIGURES_DIR}")


def run_full_pipeline():
    """Run the complete analysis pipeline."""
    print("\n" + "#"*60)
    print("# ELCI FULL PIPELINE")
    print("#"*60)

    # Step 1: Data preparation
    # Note: Uncomment if you need to fetch fresh data
    # pois, buildings = run_data_preparation()

    # Step 2: Network building
    networks = run_network_building()

    # Step 3: Accessibility calculation
    accessibility_df, tpu = run_accessibility_calculation(networks)

    # Step 4: ELCI calculation
    elci_df = run_elci_calculation(accessibility_df)

    # Step 5: Clustering analysis
    cluster_df, profiles, inertias = run_clustering_analysis(elci_df)

    # Step 6: Mismatch analysis (optional - needs population data)
    try:
        mismatch_result, ranking = run_mismatch_analysis(elci_df)
    except FileNotFoundError as e:
        print(f"\nSkipping mismatch analysis: {e}")
        mismatch_result = None

    # Step 7: Visualizations
    run_visualizations(elci_df, cluster_df, tpu)

    print("\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)


def main():
    parser = argparse.ArgumentParser(
        description="ELCI Analysis Pipeline for Hong Kong TPUs"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full pipeline"
    )
    parser.add_argument(
        "--elci-only", action="store_true",
        help="Only compute ELCI from existing accessibility data"
    )
    parser.add_argument(
        "--cluster-only", action="store_true",
        help="Only run clustering analysis"
    )
    parser.add_argument(
        "--viz-only", action="store_true",
        help="Only generate visualizations"
    )
    parser.add_argument(
        "--fetch-pois", action="store_true",
        help="Fetch POIs from OpenStreetMap"
    )
    parser.add_argument(
        "--fetch-tpu", action="store_true",
        help="Fetch TPU boundaries from HK government data"
    )

    args = parser.parse_args()

    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    if args.full:
        run_full_pipeline()
    elif args.elci_only:
        run_elci_calculation()
    elif args.cluster_only:
        run_clustering_analysis()
    elif args.viz_only:
        run_visualizations()
    elif args.fetch_pois:
        run_data_preparation(fetch_tpu=args.fetch_tpu)
    elif args.fetch_tpu:
        # Just fetch TPU boundaries
        from src.data.poi_fetcher import fetch_tpu_boundaries
        fetch_tpu_boundaries(output_path=DATA_DIR / "STPUG_21C.gpkg")
    else:
        parser.print_help()
        print("\nUse --full to run the complete pipeline.")
        print("\nIf TPU boundaries are missing, run:")
        print("  python run_pipeline.py --fetch-tpu")


if __name__ == "__main__":
    main()
