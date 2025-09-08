"""
===============================================================================
Fairness Utilities for Model Evaluation and Visualization
===============================================================================
Purpose:
    Contains helper utilities for group-based fairness evaluation in predictive
    modeling, blending Fairlearn metrics with custom visualization and comparison.

Functions:
    - plot_before_after_fairness_grid:
        Plot groupwise bar charts comparing "before" vs "after" fairness metrics.
    - plot_fairness_grid:
        Show fairness metrics (MAE, MSE, R²) for each sensitive feature/group.
    - compute_multi_feature_fairness_df:
        Compute fairness metrics for multiple sensitive features as one DataFrame.
    - compare_groupwise_fairness_disparities:
        Merge and compare before/after DataFrames for groupwise disparities.
===============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import MetricFrame





# =============================================================================
# Function: plot_before_after_fairness_grid
# =============================================================================
def plot_before_after_fairness_grid(before_df, after_df):
    """
    Plot grid comparing BEFORE (skyblue) and AFTER (lightgreen) fairness metrics
    for all groups of specified sensitive attributes.

    Parameters:
    ----------
    before_df: pd.DataFrame
        DataFrame with baseline groupwise fairness metrics ('Feature', 'Group', 'Metric', 'Value_before')

    after_df: pd.DataFrame
        DataFrame after fairness corrections ('Feature', 'Group', 'Metric', 'Value_after')

    Returns: None
    """
    print("- Plotting groupwise fairness metrics (before vs after)...", end=" ")

    # Merge 'before' and 'after' DataFrames
    merged_df = pd.merge(
        before_df,
        after_df,
        on=['Feature', 'Group', 'Metric'],
        suffixes=('_before', '_after')
    )

    # Melt to long form for easier grouped plotting
    plot_df = merged_df.melt(
        id_vars=['Feature', 'Group', 'Metric'],
        value_vars=['Value_before', 'Value_after'],
        var_name='Stage',
        value_name='Value'
    )

    # Standardize Stage to clean labels
    plot_df['Stage'] = plot_df['Stage'].map({
        'Value_before': 'Before',
        'Value_after': 'After'
    })

    # Find unique metrics and features for subplots
    sensitive_features = plot_df['Feature'].unique()
    metrics = plot_df['Metric'].unique()
    n_features = len(sensitive_features)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_features, n_metrics,
                             figsize=(n_metrics * 5, n_features * 4),
                             squeeze=False)
    fig.suptitle('Before (skyblue) vs After (lightgreen) Fairness Metrics', fontsize=20, y=1.05)
    for i, feature in enumerate(sensitive_features):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            subset = plot_df[(plot_df['Feature'] == feature) & (plot_df['Metric'] == metric)]
            sns.barplot(
                data=subset,
                x='Group',
                y='Value',
                hue='Stage',
                palette={'Before': 'skyblue', 'After': 'lightgreen'},
                ax=ax
            )
            ax.set_title(f'{metric} by {feature}')
            ax.set_xlabel('Group')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print("done.")





# =============================================================================
# Function: plot_fairness_grid
# =============================================================================
def plot_fairness_grid(fairness_df):
    """
    Show grid of bar charts for fairness metrics across features/groups.

    Parameters:
    ----------
    fairness_df: pd.DataFrame
        DataFrame containing 'Feature', 'Group', 'Metric', 'Value'

    Returns: None
    """
    print("- Plotting fairness metrics grid...", end=" ")

    sensitive_features = fairness_df['Feature'].unique()
    metrics = fairness_df['Metric'].unique()
    n_features = len(sensitive_features)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_features, n_metrics,
                             figsize=(n_metrics * 5, n_features * 4),
                             squeeze=False)
    fig.suptitle('Group-wise Fairness Metrics Grid', fontsize=20, y=1.03)
    for i, feature in enumerate(sensitive_features):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            plot_data = fairness_df[
                (fairness_df['Feature'] == feature) &
                (fairness_df['Metric'] == metric)
            ]
            sns.barplot(data=plot_data, x='Group', y='Value', ax=ax, palette='viridis')
            ax.set_title(f'{metric} by {feature}')
            ax.set_xlabel('Group')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    print("done.")





# =============================================================================
# Function: compute_multi_feature_fairness_df
# =============================================================================
def compute_multi_feature_fairness_df(y_true, y_pred, df_features, sensitive_features, metrics_dict):
    """
    Compute groupwise fairness metrics for multiple sensitive features.

    Parameters:
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    df_features : pd.DataFrame
        DataFrame with all sensitive features as columns.
    sensitive_features : list of str
        Names of features to evaluate fairness on.
    metrics_dict : dict
        Metric names to Fairlearn or sklearn metric functions.

    Returns:
    -------
    pd.DataFrame
        Groupwise fairness metric results with columns ['Feature', 'Group', 'Metric', 'Value']
    """
    print("- Computing fairness metrics per sensitive feature...", end=" ")
    all_results = []
    for feature in sensitive_features:
        metric_frame = MetricFrame(
            metrics=metrics_dict,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=df_features[feature]
        )
        # Ensure that the group column is named 'Group'
        df_grouped = metric_frame.by_group.reset_index().rename(columns={feature: 'Group'})
        # Melt so each row has a metric name and value
        df_melted = df_grouped.melt(
            id_vars='Group',
            var_name='Metric',
            value_name='Value'
        )
        df_melted['Feature'] = feature
        all_results.append(df_melted)
    final_df = pd.concat(all_results, ignore_index=True)
    print("done.")
    return final_df[['Feature', 'Group', 'Metric', 'Value']]





# =============================================================================
# Function: compare_groupwise_influence
# =============================================================================
def compare_groupwise_influence(before_df, after_df):
    """
    Merge & compare 'before' and 'after' groupwise influence DataFrames.

    Parameters:
    ----------
    before_df: pd.DataFrame
        DataFrame with baseline metrics.
    after_df: pd.DataFrame
        DataFrame after influence corrections.

    Returns:
    -------
    pd.DataFrame
        DataFrame including improvement and its percent change for each group.
    """
    print("- Comparing before/after metrics for each group (influence)...", end=" ")
    merged_df = pd.merge(
        before_df,
        after_df,
        on=['Feature', 'Group', 'Metric'],
        suffixes=('_before', '_after')
    )

    # Compute improvement: error metrics (MAE/MSE) are improved if decreased;
    # non-error metrics (e.g., R²) are improved if increased
    is_error_metric = merged_df['Metric'].isin(['MAE', 'MSE'])

    merged_df.loc[is_error_metric, 'Improvement'] = \
        merged_df['Value_before'] - merged_df['Value_after']

    merged_df.loc[~is_error_metric, 'Improvement'] = \
        merged_df['Value_after'] - merged_df['Value_before']

    merged_df['Improvement (%)'] = (merged_df['Improvement'] / abs(merged_df['Value_before'])) * 100
    print("done.")
    return merged_df[['Feature', 'Group', 'Metric', 'Value_before', 'Value_after', 'Improvement', 'Improvement (%)']]
