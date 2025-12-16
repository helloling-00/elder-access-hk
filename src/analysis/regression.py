# -*- coding: utf-8 -*-
"""
Regression analysis for elderly population and facility distribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def predict_elderly_population_trend(
    df: pd.DataFrame,
    population_cols: List[str],
    years: List[int],
    predict_year: int,
    tpu_id_col: str = "stpug_eng"
) -> pd.DataFrame:
    """
    Predict elderly population for future year using linear regression.

    Args:
        df: DataFrame with historical population data
        population_cols: Columns with population counts for each year
        years: List of years corresponding to population_cols
        predict_year: Year to predict
        tpu_id_col: TPU identifier column

    Returns:
        DataFrame with predicted population column
    """
    df = df.copy()

    years_arr = np.array(years).reshape(-1, 1)
    predictions = []

    print(f"Predicting elderly population for {predict_year}...")

    for _, row in df.iterrows():
        pops = row[population_cols].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(years_arr, pops)

        pred = model.predict([[predict_year]])[0][0]
        predictions.append(max(0, pred))  # Ensure non-negative

    output_col = f"POP65PLUS_{predict_year}_pred"
    df[output_col] = predictions

    print(f"Prediction complete. Column: {output_col}")
    print(f"Mean predicted population: {np.mean(predictions):.1f}")

    return df


def analyze_facility_population_relationship(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    model_type: str = "linear"
) -> Dict[str, Any]:
    """
    Analyze relationship between facility counts and elderly population.

    Args:
        df: DataFrame with population and facility data
        dependent_var: Target variable (e.g., elderly population)
        independent_vars: Predictor variables (e.g., facility counts)
        model_type: "linear", "random_forest", or "xgboost"

    Returns:
        Dictionary with model results and metrics
    """
    # Prepare data
    valid_df = df[independent_vars + [dependent_var]].dropna()
    X = valid_df[independent_vars]
    y = valid_df[dependent_var]

    print(f"Analyzing with {len(valid_df)} valid samples...")

    # Select model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fit model
    model.fit(X, y)
    y_pred = model.predict(X)

    # Compute metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    results = {
        "model": model,
        "model_type": model_type,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "predictions": y_pred,
        "actual": y.values,
    }

    # Feature importance for tree-based models
    if hasattr(model, "feature_importances_"):
        results["feature_importance"] = dict(
            zip(independent_vars, model.feature_importances_)
        )

    # Coefficients for linear models
    if hasattr(model, "coef_"):
        results["coefficients"] = dict(
            zip(independent_vars, model.coef_)
        )
        results["intercept"] = model.intercept_

    print(f"\n{model_type.upper()} Model Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


def compare_models(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple regression models.

    Args:
        df: Input DataFrame
        dependent_var: Target variable
        independent_vars: Predictor variables

    Returns:
        Dictionary mapping model names to results
    """
    models = {}

    for model_type in ["linear", "random_forest"]:
        print(f"\n{'='*50}")
        print(f"Fitting {model_type} model...")
        models[model_type] = analyze_facility_population_relationship(
            df, dependent_var, independent_vars, model_type
        )

    # Summary comparison
    print(f"\n{'='*50}")
    print("MODEL COMPARISON:")
    print("-" * 50)
    print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'CV R²':<10}")
    print("-" * 50)

    for name, results in models.items():
        print(f"{name:<15} {results['r2']:<10.4f} {results['rmse']:<10.2f} {results['cv_mean']:<10.4f}")

    return models


class RegressionAnalyzer:
    """
    Class to manage regression analysis workflow.
    """

    def __init__(self):
        """Initialize regression analyzer."""
        self.models: Dict[str, Dict[str, Any]] = {}
        self.predictions: Optional[pd.DataFrame] = None

    def predict_population_trend(
        self,
        df: pd.DataFrame,
        population_cols: List[str],
        years: List[int],
        predict_year: int
    ) -> pd.DataFrame:
        """
        Predict future elderly population.

        Args:
            df: Historical population data
            population_cols: Population columns
            years: Years for each column
            predict_year: Target prediction year

        Returns:
            DataFrame with predictions
        """
        self.predictions = predict_elderly_population_trend(
            df, population_cols, years, predict_year
        )
        return self.predictions

    def fit_models(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit and compare regression models.

        Args:
            df: Input DataFrame
            dependent_var: Target variable
            independent_vars: Predictor variables

        Returns:
            Model comparison results
        """
        self.models = compare_models(df, dependent_var, independent_vars)
        return self.models

    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model.

        Returns:
            Tuple of (model name, model results)
        """
        if not self.models:
            raise ValueError("Must fit models first")

        best_name = max(self.models, key=lambda x: self.models[x]["cv_mean"])
        return best_name, self.models[best_name]
