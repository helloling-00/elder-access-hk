# -*- coding: utf-8 -*-
"""
Spatial clustering analysis for TPU accessibility patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from ..config import DEFAULT_N_CLUSTERS


DEFAULT_CLUSTER_FEATURES = [
    "transit_walk_min_norm",
    "hospital_walk_min_norm",
    "hospital_drive_min_norm",
    "daily_walk_min_norm",
    "service_walk_min_norm",
    "leisure_walk_min_norm",
]


def prepare_clustering_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    impute_strategy: str = "median"
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare data for clustering by handling missing values.

    Args:
        df: Input DataFrame
        feature_cols: Columns to use for clustering
        impute_strategy: Strategy for imputing missing values

    Returns:
        Tuple of (imputed array, feature DataFrame)
    """
    # Extract features
    X = df[feature_cols].copy()

    # Report missing values
    missing = X.isna().sum()
    if missing.sum() > 0:
        print("Missing values per column:")
        print(missing[missing > 0])

    # Impute missing values
    imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)

    return X_imputed, X


def kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    output_col: str = "cluster_kmeans",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform K-means clustering on TPU accessibility data.

    Args:
        df: DataFrame with normalized accessibility features
        feature_cols: Columns to use for clustering
        n_clusters: Number of clusters
        output_col: Output column name for cluster labels
        random_state: Random seed

    Returns:
        DataFrame with cluster labels added
    """
    df = df.copy()
    feature_cols = feature_cols or DEFAULT_CLUSTER_FEATURES

    # Prepare data
    X_imputed, _ = prepare_clustering_data(df, feature_cols)

    # Perform clustering
    print(f"Performing K-means clustering with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df[output_col] = km.fit_predict(X_imputed)

    # Print cluster sizes
    print("\nCluster sizes:")
    print(df[output_col].value_counts().sort_index())

    return df


def hierarchical_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    method: str = "ward",
    output_col: str = "cluster_hier"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Perform hierarchical clustering on TPU data.

    Args:
        df: DataFrame with normalized accessibility features
        feature_cols: Columns to use for clustering
        n_clusters: Number of clusters to cut
        method: Linkage method
        output_col: Output column name for cluster labels

    Returns:
        Tuple of (DataFrame with labels, linkage matrix)
    """
    df = df.copy()
    feature_cols = feature_cols or DEFAULT_CLUSTER_FEATURES

    # Prepare data
    X_imputed, _ = prepare_clustering_data(df, feature_cols)

    # Compute linkage
    print(f"Performing hierarchical clustering with method={method}...")
    Z = linkage(X_imputed, method=method)

    # Cut tree
    df[output_col] = fcluster(Z, t=n_clusters, criterion="maxclust")

    print("\nCluster sizes:")
    print(df[output_col].value_counts().sort_index())

    return df, Z


def compute_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute mean feature values for each cluster.

    Args:
        df: DataFrame with cluster labels
        cluster_col: Column containing cluster labels
        feature_cols: Feature columns to profile

    Returns:
        DataFrame with cluster profiles (rows=clusters, cols=features)
    """
    feature_cols = feature_cols or DEFAULT_CLUSTER_FEATURES
    valid_cols = [c for c in feature_cols if c in df.columns]

    profile = df.groupby(cluster_col)[valid_cols].mean()

    print("\nCluster Profiles:")
    print(profile.round(3))

    return profile


def find_optimal_k(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> List[float]:
    """
    Find optimal number of clusters using elbow method.

    Args:
        df: DataFrame with features
        feature_cols: Columns to use
        k_range: Range of k values to test
        random_state: Random seed

    Returns:
        List of inertia values for each k
    """
    feature_cols = feature_cols or DEFAULT_CLUSTER_FEATURES
    X_imputed, _ = prepare_clustering_data(df, feature_cols)

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_imputed)
        inertias.append(km.inertia_)
        print(f"k={k}: inertia={km.inertia_:.2f}")

    return inertias


class ClusterAnalyzer:
    """
    Class to manage clustering analysis workflow.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        n_clusters: int = DEFAULT_N_CLUSTERS
    ):
        """
        Initialize cluster analyzer.

        Args:
            feature_cols: Feature columns for clustering
            n_clusters: Default number of clusters
        """
        self.feature_cols = feature_cols or DEFAULT_CLUSTER_FEATURES
        self.n_clusters = n_clusters
        self.result: Optional[pd.DataFrame] = None
        self.profiles: Optional[pd.DataFrame] = None
        self.linkage_matrix: Optional[np.ndarray] = None

    def fit_kmeans(
        self,
        df: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fit K-means clustering.

        Args:
            df: Input DataFrame
            n_clusters: Number of clusters (uses default if None)

        Returns:
            DataFrame with cluster labels
        """
        n_clusters = n_clusters or self.n_clusters
        self.result = kmeans_clustering(
            df, self.feature_cols, n_clusters
        )
        return self.result

    def fit_hierarchical(
        self,
        df: pd.DataFrame,
        n_clusters: Optional[int] = None,
        method: str = "ward"
    ) -> pd.DataFrame:
        """
        Fit hierarchical clustering.

        Args:
            df: Input DataFrame
            n_clusters: Number of clusters
            method: Linkage method

        Returns:
            DataFrame with cluster labels
        """
        n_clusters = n_clusters or self.n_clusters
        self.result, self.linkage_matrix = hierarchical_clustering(
            df, self.feature_cols, n_clusters, method
        )
        return self.result

    def get_profiles(
        self,
        cluster_col: str = "cluster_kmeans"
    ) -> pd.DataFrame:
        """
        Get cluster profiles.

        Args:
            cluster_col: Cluster label column

        Returns:
            Cluster profiles DataFrame
        """
        if self.result is None:
            raise ValueError("Must fit clustering first")

        self.profiles = compute_cluster_profiles(
            self.result, cluster_col, self.feature_cols
        )
        return self.profiles

    def find_optimal_k(
        self,
        df: pd.DataFrame,
        k_range: range = range(2, 9)
    ) -> List[float]:
        """
        Find optimal k using elbow method.

        Args:
            df: Input DataFrame
            k_range: Range of k values

        Returns:
            List of inertia values
        """
        return find_optimal_k(df, self.feature_cols, k_range)
