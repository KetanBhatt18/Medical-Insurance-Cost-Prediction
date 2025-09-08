"""
===========================================================
Stratified Regression Analysis Utilities
===========================================================

This module provides functions for:
- Performing stratified train-test splits on a DataFrame
- Training regression models with preserved subgroup representation
- Evaluating model performance using RMSE, MAE, and R²
- Visualizing predictions and feature importances

Core Features:
--------------
- `get_stratified_split()`  → Splits dataset while maintaining proportions 
  of a specified categorical column (useful for fairness-sensitive tasks)
- `evaluate_model()`        → Computes both training and test metrics
- `plot_predictions()`      → Shows predicted vs actual values with regression line
- `plot_feature_importance()` → Displays coefficient or feature importance ranking
- `run_stratified_regression()` → Orchestrates the full pipeline from split to visualization

Intended Use:
-------------
Designed for regression tasks where preservation of subgroup representation 
(e.g., by risk profile, demographic group, etc.) is important for 
both baseline model performance and fairness analysis.

Author: [Your Name / Team]
Date:   August 2025
===========================================================
"""

# =========================
# Data Handling
# =========================
import pandas as pd
import numpy as np

# =========================
# Visualization
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Modeling and preprocessing
# =========================
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression

# Example model support
from xgboost import XGBRegressor


def get_stratified_split(df, target='charges', stratify_by='risk_profile_id',
                         test_size=0.2, random_state=42):
    """
    Perform a stratified train-test split based on a categorical grouping column.

    This ensures that both the training and test sets maintain the proportional 
    representation of the specified `stratify_by` column.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing features and target.
    target : str, default='charges'
        Name of target column to predict.
    stratify_by : str
        Column to stratify splits on.
    test_size : float, default=0.2
        Proportion of data to assign to test set.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target values.
    X_test : pd.DataFrame
        Test feature set.
    y_test : pd.Series
        Test target values.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    # The stratification ensures subgroup distribution is preserved
    for train_idx, test_idx in splitter.split(df, df[stratify_by]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return X_train, y_train, X_test, y_test


def get_scores(y_train, y_pred_train, y_test, y_pred_test):
    """
    Compute regression performance metrics for training and test sets.

    Returns
    -------
    dict
        Dictionary with rmse_train, rmse_test, mae_train, mae_test,
        r2_train, r2_test.
    """
    scores = {
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test":  np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mae_train":  mean_absolute_error(y_train, y_pred_train),
        "mae_test":   mean_absolute_error(y_test, y_pred_test),
        "r2_train":   r2_score(y_train, y_pred_train),
        "r2_test":    r2_score(y_test, y_pred_test)
    }
    return scores


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Generate performance metrics for a trained model.

    Parameters
    ----------
    model : estimator
        A scikit-learn or compatible regression model.
    X_train, X_test : pd.DataFrame
        Training and test features.
    y_train, y_test : pd.Series
        Ground truth target values.

    Returns
    -------
    scores : dict
        Regression metrics for train and test sets.
    y_pred_train : np.ndarray
        Model predictions for training set.
    y_pred_test : np.ndarray
        Model predictions for test set.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    scores = get_scores(y_train, y_pred_train, y_test, y_pred_test)
    return scores, y_pred_train, y_pred_test


def plot_predictions(y_train, y_pred_train, y_test, y_pred_test):
    """
    Visualize predicted vs actual scatter plots for train and test sets.

    The plot includes:
    - Scatter points showing predictions
    - Best-fit regression line
    - Reference line y = x for perfect predictions
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def plot_fit(ax, y_true, y_pred, title, color, line_color):
        # Fit a simple line to measure prediction bias
        fit = LinearRegression().fit(y_true.values.reshape(-1, 1), y_pred)
        line = fit.predict(y_true.values.reshape(-1, 1))
        slope = fit.coef_[0]
        intercept = fit.intercept_

        # Scatter points + regression line
        ax.scatter(y_true, y_pred, alpha=0.2, color=color)
        ax.plot(y_true, line, color=line_color, label='Best Fit')

        # Reference perfect-prediction line
        ref_line = np.linspace(y_true.min(), y_true.max(), 100)
        ax.plot(ref_line, ref_line, linestyle='--', color='gray', label='y = x')

        ax.set_title(title)
        ax.set_xlabel("Actual Charges")
        ax.legend()
        return slope, intercept

    s_train, i_train = plot_fit(
        axes[0], y_train, y_pred_train,
        "Training: Predictions vs Actual", 'skyblue', 'navy'
    )
    s_test, i_test = plot_fit(
        axes, y_test, y_pred_test,
        "Test: Predictions vs Actual", 'coral', 'darkred'
    )

    print(f"\n📐 Training Best-Fit Line: y = {s_train:.4f}x + {i_train:.2f}")
    print(f"📐 Test Best-Fit Line: y = {s_test:.4f}x + {i_test:.2f}")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, X_train):
    """
    Display model feature importance (coefficients or tree-based importance).

    Works for:
    - Linear models with `.coef_`
    - Tree-based models with `.feature_importances_`
    """
    if hasattr(model, 'coef_'):
        importances = pd.Series(model.coef_, index=X_train.columns)
    elif hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
    else:
        print("\n⚠️ Feature importance not available for this model.")
        return

    importances.sort_values().plot(
        kind='barh', figsize=(8, 6), color='teal'
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def run_stratified_regression(df, model, target='charges',
                               stratify_by='risk_profile_id',
                               test_size=0.2, random_state=42):
    """
    Complete pipeline: split → train → evaluate → plot.

    Steps:
    ------
    1. Stratified split to preserve subgroup proportions.
    2. Fit user-supplied regression model.
    3. Evaluate and print RMSE, MAE, R² for train/test sets.
    4. Visualize predictions and feature importance.

    Returns
    -------
    Trained model instance.
    """
    # Step 1: Data split
    X_train, y_train, X_test, y_test = get_stratified_split(
        df, target=target, stratify_by=stratify_by, test_size=test_size
    )

    # Step 2: Model training
    model.fit(X_train, y_train)

    # Step 3: Evaluation
    scores, y_pred_train, y_pred_test = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    # Print performance table
    performance = pd.DataFrame({
        'RMSE': [scores["rmse_train"], scores["rmse_test"]],
        'MAE':  [scores["mae_train"],  scores["mae_test"]],
        'R²':   [scores["r2_train"],   scores["r2_test"]]
    }, index=['Training', 'Test'])
    print("\n📊 Model Performance:\n", performance.round(4))

    # Step 4: Visual outputs
    plot_predictions(y_train, y_pred_train, y_test, y_pred_test)
    plot_feature_importance(model, X_train)

    return model
