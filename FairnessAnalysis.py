"""
===========================================================
Dataset Influence Analysis Utilities
===========================================================

This module provides a set of utility functions to evaluate 
group-based influence metrics across categorical and numeric 
features, focusing on differences in target variable behavior.
It includes:

- Groupwise influence computation (mean & median based)
- Subgroup imbalance calculation
- Binary feature target influence comparison
- Feature skewness analysis
- Markdown export utilities for influence reporting
- Influence distribution visualization (KDE & Boxplots)

Intended Use:
-------------
These functions are primarily designed to support fairness,
bias, and distributional influence audits on a dataset where 
certain features (protected or otherwise) may have measurable
impact on the target variable distribution.

Author: [Your Name / Team]
Date: August 2025
Dependencies:
    pandas, numpy, matplotlib, seaborn, joblib
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================
import pandas as pd
import numpy as np
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from IPython.display import Markdown

# ===========================================================
# Influence calculation and related functions
# ===========================================================

# This function provides an imbalanced representation of any categorical attribute in the dataset
def expand_groupwise_influence(df, feature_col, target_col):
    """
    Computes groupwise mean and delta vs overall mean for a non-binary feature.
    Returns a clean DataFrame ready for markdown export.
    
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - feature_col (str): The categorical feature column to group by.
    - target_col (str): The numeric target column for mean calculation.
    
    Returns:
    - pd.DataFrame: DataFrame with group means and differences from overall mean.
    """
    overall_mean = df[target_col].mean()
    rows = []
    for val in sorted(df[feature_col].dropna().unique()):
        group_df = df[df[feature_col] == val]
        group_mean = group_df[target_col].mean()
        delta = group_mean - overall_mean
        rows.append({
            "Feature": feature_col,
            "Group": val,
            "Group Mean": round(group_mean, 2),
            "Δ from Overall Mean": round(delta, 2)
        })
    return pd.DataFrame(rows)


def pivot_influences(df_influence):
    """
    Reshapes the output for each feature into a wide table format:
    Feature | Mean_Group_A | Mean_Group_B | ... | Δ_Max
    
    Parameters:
    - df_influence (pd.DataFrame): Long format influences DataFrame.
    
    Returns:
    - pd.DataFrame: Pivoted wide format influences table with delta max-column.
    """
    tables = []
    for feature in df_influence['Feature'].unique():
        sub_df = df_influence[df_influence['Feature'] == feature]
        pivot_row = {"Feature": feature}
        deltas = []
        for _, row in sub_df.iterrows():
            col_name = f"Mean_{row['Group']}"
            pivot_row[col_name] = row["Group Mean"]
            deltas.append(row["Δ from Overall Mean"])
        pivot_row["Δ_Max"] = round(max(deltas) - min(deltas), 2)
        tables.append(pivot_row)
    return pd.DataFrame(tables)


def export_target_influence_markdown(df_influence, label=None):
    """
    Converts influences into markdown for reporting.
    Can be used on raw (long) or pivoted (wide) format.
    
    Parameters:
    - df_influence (pd.DataFrame): Influence DataFrame to convert.
    - label (str, optional): Section label for markdown.
    
    Returns:
    - str: Markdown string representation of influences.
    """
    md = ""
    if label:
        md += f"### {label}\n\n"
    md += df_influence.to_markdown(index=False)
    return md


def subgroup_imbalance(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calculates imbalance ratios for categorical subgroups.
    
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - features (list): List of categorical features to check.
    
    Returns:
    - pd.DataFrame: Each feature with value counts and imbalance ratio.
    """
    result = []
    for feature in features:
        if feature not in df.columns:
            result.append({
                "Feature": feature,
                "Value Counts": "❌ Not found",
                "Imbalance Ratio": "—"
            })
            continue
        counts = df[feature].value_counts()
        ratio = round(counts.min() / counts.max(), 3) if len(counts) > 1 else 1.0
        result.append({
            "Feature": feature,
            "Value Counts": dict(counts),
            "Imbalance Ratio": ratio
        })
    return pd.DataFrame(result)


def target_influences(df: pd.DataFrame, target: str, features: list) -> pd.DataFrame:
    """
    Computes target mean influences between two groups for binary features.
    
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - target (str): Target variable column name.
    - features (list): List of feature names to analyze.
    
    Returns:
    - pd.DataFrame: Mean values and delta for each binary feature.
    """
    result = []
    for feature in features:
        if feature not in df.columns:
            result.append({
                "Feature": feature,
                "Group A Mean": "❌ Not found",
                "Group B Mean": "—",
                "Δ Mean": "—"
            })
            continue
        group_stats = df.groupby(feature)[target].agg(['mean', 'count'])
        if group_stats.shape[0] == 2:
            delta = abs(group_stats.iloc['mean'] - group_stats.iloc['mean'])
            result.append({
                "Feature": feature,
                "Group A Mean": round(group_stats.iloc['mean'], 2),
                "Group B Mean": round(group_stats.iloc['mean'], 2),
                "Δ Mean": round(delta, 2)
            })
        else:
            result.append({
                "Feature": feature,
                "Group A Mean": "⚠️ Not binary",
                "Group B Mean": "—",
                "Δ Mean": "—"
            })
    return pd.DataFrame(result)


def feature_skew(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Computes skewness for numeric features and flags highly skewed ones.
    
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - features (list): List of feature names to analyze.
    
    Returns:
    - pd.DataFrame: Feature skewness and highly skewed flag.
    """
    result = []
    for feature in features:
        if feature not in df.columns:
            result.append({
                "Feature": feature,
                "Skewness": "❌ Not found",
                "Highly Skewed": "—"
            })
            continue
        if pd.api.types.is_numeric_dtype(df[feature]):
            skew = round(df[feature].skew(), 2)
            result.append({
                "Feature": feature,
                "Skewness": skew,
                "Highly Skewed": "Yes" if abs(skew) > 1 else "No"
            })
        else:
            result.append({
                "Feature": feature,
                "Skewness": "⚠️ Not numeric",
                "Highly Skewed": "—"
            })
    return pd.DataFrame(result)


def Dataset_Fairness_Evaluation(df: pd.DataFrame, target: str, fairness_tags: list) -> Markdown:
    """
    Generates a markdown fairness profile summary using mean-based influence metrics.
    
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - target (str): Target variable column name.
    - fairness_tags (list): List of feature names related to fairness.
    
    Returns:
    - IPython.display.Markdown: Markdown formatted fairness summary.
    """
    imbalance_df = subgroup_imbalance(df, fairness_tags)
    influence_df = target_influences(df, target, fairness_tags)
    skew_df = feature_skew(df, fairness_tags)
    markdown = "## Dataset Fairness Profile Summary\n\n"
    markdown += "### Subgroup Imbalance\n"
    markdown += "| Feature | Value Counts | Imbalance Ratio |\n"
    markdown += "|---------|--------------|-----------------|\n"
    for _, row in imbalance_df.iterrows():
        markdown += f"| `{row['Feature']}` | {row['Value Counts']} | {row['Imbalance Ratio']} |\n"
    markdown += "\n### Target Influences Across Groups\n"
    markdown += "| Feature | Group A Mean | Group B Mean | Δ Mean |\n"
    markdown += "|---------|--------------|--------------|--------|\n"
    for _, row in influence_df.iterrows():
        markdown += f"| `{row['Feature']}` | {row['Group A Mean']} | {row['Group B Mean']} | {row['Δ Mean']} |\n"
    markdown += "\n### Feature Skewness\n"
    markdown += "| Feature | Skewness | Highly Skewed |\n"
    markdown += "|---------|----------|---------------|\n"
    for _, row in skew_df.iterrows():
        markdown += f"| `{row['Feature']}` | {row['Skewness']} | {row['Highly Skewed']} |\n"
    return Markdown(markdown)


def expand_groupwise_influence_median(df, feature_col, target_col):
    """
    Computes groupwise median and delta vs overall median for a non-binary feature.
    Returns a clean DataFrame with overall median included for markdown export.
    
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - feature_col (str): The categorical feature column to group by.
    - target_col (str): The numeric target column for median calculation.
    
    Returns:
    - pd.DataFrame: DataFrame with group medians, overall median, deltas, and group sizes.
    """
    overall_median = df[target_col].median()
    rows = []
    for val in sorted(df[feature_col].dropna().unique()):
        group_df = df[df[feature_col] == val]
        group_median = group_df[target_col].median()
        delta = group_median - overall_median
        rows.append({
            "Feature": feature_col,
            "Group": val,
            "Group Median": round(group_median, 2),
            "Overall Median": round(overall_median, 2),
            "Δ vs Overall Median": round(delta, 2),
            "N": len(group_df)
        })
    return pd.DataFrame(rows)


def target_influences_by_median(df: pd.DataFrame, target: str, features: list) -> pd.DataFrame:
    """
    Computes groupwise median and delta vs overall median for a list of features.
    Includes overall median for each row to support audit context.
    """
    rows = []
    overall_median = df[target].median()
    for feature in features:
        if feature not in df.columns:
            rows.append({
                "Feature": feature,
                "Group": "❌",
                "Group Median": None,
                "Overall Median": round(overall_median, 2),
                "Δ vs Overall Median": None
            })
            continue
        group_stats = df.groupby(feature)[target].median().reset_index()
        group_stats.columns = ["Group", "Group Median"]
        group_stats["Overall Median"] = round(overall_median, 2)
        group_stats["Δ vs Overall Median"] = group_stats["Group Median"].apply(lambda x: round(x - overall_median, 2))
        group_stats["Feature"] = feature
        rows.extend(group_stats.to_dict("records"))
    return pd.DataFrame(rows)[["Feature", "Group", "Group Median", "Overall Median", "Δ vs Overall Median"]]


def Dataset_Fairness_Evaluation_Median(df: pd.DataFrame, target: str, fairness_tags: list) -> Markdown:
    """
    Generates a markdown fairness profile summary using median-based influence metrics.
    """
    imbalance_df = subgroup_imbalance(df, fairness_tags)
    influence_df = target_influences_by_median(df, target, fairness_tags)
    skew_df = feature_skew(df, fairness_tags)
    markdown = "## Dataset Fairness Profile Summary (Median Based)\n\n"
    markdown += "### Subgroup Imbalance\n"
    markdown += imbalance_df.to_markdown(index=False) + "\n\n"
    markdown += "### Target Influences (vs Overall Median)\n"
    markdown += influence_df.to_markdown(index=False) + "\n\n"
    markdown += "### Feature Skewness\n"
    markdown += skew_df.to_markdown(index=False) + "\n"
    return Markdown(markdown)


def plot_groupwise_influence_overlays(df, feature_col, target_col, overall_median=None, figsize=(10, 4)):
    """
    Plots KDE and boxplot overlays per category of a non-binary feature,
    comparing distributions to overall target median.
    """
    groups = sorted(df[feature_col].dropna().unique())
    n_groups = len(groups)
    overall_median = overall_median or df[target_col].median()
    fig, axes = plt.subplots(n_groups, 2, figsize=(figsize[0], figsize * n_groups))
    if n_groups == 1:
        axes = [axes]  # Handle edge case of 1 group
    for idx, group in enumerate(groups):
        group_df = df[df[feature_col] == group]
        # KDE Overlay
        sns.kdeplot(group_df[target_col], ax=axes[idx], fill=True, linewidth=1.5, label=str(group))
        axes[idx].axvline(overall_median, color='red', linestyle='--', label='Overall Median')
        axes[idx].axvline(group_df[target_col].median(), color='blue', linestyle='-', label='Group Median')
        axes[idx].legend()
        axes[idx].set_title(f"{feature_col} = {group} | KDE")
        # Boxplot Overlay
        sns.boxplot(x=group_df[target_col], ax=axes[idx], color='skyblue')
        axes[idx].axvline(overall_median, color='red', linestyle='--', label='Overall Median')
        axes[idx].axvline(group_df[target_col].median(), color='blue', linestyle='-', label='Group Median')
        axes[idx].legend()
        axes[idx].set_title(f"{feature_col} = {group} | Boxplot")
    plt.tight_layout()
    plt.show()

