"""
===========================================================
Fairness-Aware Data Transformation Utilities
===========================================================

This module provides preprocessing and transformation utilities for 
tabular datasets, especially in the context of fairness-aware 
machine learning workflows.

Core Capabilities:
------------------
1. Data balancing using SMOTE / SMOTENC
2. Quantile transformation of continuous features
3. Generation of polynomial/interaction features
4. Binning of continuous variables into categorical bins
5. Behavioral flag creation from domain knowledge, now including smoker and non-smoker flags and their interactions
6. Feature selection based on correlation thresholds
7. Categorical column combination and fuzzy-mapping utilities
8. Fairness-oriented upsampling with numeric perturbations
9. Complete fairness-aware transformation pipeline (`transform_data`)

Intended Use:
-------------
- Equalizing representation across demographic or behavioral subgroups
- Creating additional engineered features to improve fairness audits
- Normalizing distributions and removing low-relevance features
- Generating consistent transformations for model pipelines

Dependencies:
-------------
pandas, numpy, seaborn, matplotlib, sklearn, joblib, imblearn, IPython

Author:
-------
[Your Name / Team]
Date: August 2025
===========================================================
"""

# ===========
# Imports
# ===========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import itertools
from difflib import get_close_matches
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE, SMOTENC
from IPython.display import Markdown
from scipy.stats import truncnorm


warnings.filterwarnings("ignore", category=FutureWarning)


# ===========
# Function: Upsample using SMOTE (numeric features only)
# ===========
def upsample_with_smote(data, y, categorical_features=None):
    """
    Balance dataset using SMOTE or SMOTENC depending on feature types.

    If `categorical_features` is provided, uses SMOTENC for mixed-type data. 
    Otherwise, uses standard SMOTE, which works only on numeric features.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and target.
    y : str
        Target column name to balance.
    categorical_features : list, optional
        List of categorical feature column names (for SMOTENC).

    Returns
    -------
    pd.DataFrame
        Balanced dataset after upsampling.
    """
    df = data.copy()

    # Choose appropriate SMOTE variant
    if categorical_features:
        cat_indices = [df.columns.get_loc(col)
                       for col in categorical_features if col in df.columns]
        smote = SMOTENC(categorical_features=cat_indices, random_state=42)
    else:
        smote = SMOTE(random_state=42)

    print(f"\nUpsampling dataset to balance classes in target '{y}'\n")

    # Before upsampling visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    cp1 = sns.countplot(data=df, x=y)
    #cp1.bar_label(cp1.containers[0])
    plt.title('Original Data')

    # Fit-resample features and target
    features = df.drop(columns=[y])
    target = df[y]
    features_res, target_res = smote.fit_resample(features, target)

    df_res = pd.concat([features_res, target_res], axis=1)

    # After upsampling visualization
    plt.subplot(1, 2, 2)
    cp2 = sns.countplot(data=df_res, x=y)
    #cp2.bar_label(cp2.containers)
    plt.title('After Upsampling')
    plt.show()

    return df_res


# ===========
# Function: Quantile Transformation
# ===========
def quantile_transform(df, columns, output_suffix='_qt', n_quantiles=1000, output_distribution='uniform'):
    """
    Apply sklearn QuantileTransformer on selected continuous columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list
        Columns to transform.
    output_suffix : str, default='_qt'
        Suffix to append to new transformed features.
    n_quantiles : int, default=1000
        Number of quantiles for mapping.
    output_distribution : {'uniform', 'normal'}, default='uniform'
        Target distribution of transformed data.

    Returns
    -------
    pd.DataFrame
        Dataset with additional quantile-transformed columns.
    """
    df_trans = df.copy()
    qt = QuantileTransformer(n_quantiles=n_quantiles,
                             output_distribution=output_distribution,
                             random_state=42)

    for col in columns:
        reshaped = df[[col]].values
        transformed = qt.fit_transform(reshaped)
        df_trans[col + output_suffix] = transformed.flatten()

    # Save fitted transformer for pipeline reproducibility
    joblib.dump(qt, 'quantile_transformer_model.pkl')
    return df_trans


# ===========
# Function: Generate Interaction Features
# ===========
def generate_interactions(df, attributes, max_order=3, delimiter='__'):
    """
    Generate polynomial-style interaction features from base attributes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    attributes : list
        Column names to combine.
    max_order : int, default=3
        Maximum combination order (2 = pairs, 3 = triples, etc.).
    delimiter : str, default='__'
        String to use when naming new features.

    Returns
    -------
    pd.DataFrame
        Dataset with additional interaction features.
    """
    df_interact = df.copy()
    for r in range(2, max_order + 1):
        combos = itertools.combinations(attributes, r)
        for combo in combos:
            col_name = delimiter.join(combo)
            product = 1
            for c in combo:
                # Fill NaNs to avoid multiplication errors
                product *= df_interact[c].fillna(0)
            df_interact[col_name] = product
    return df_interact


# ===========
# Function: Bin Continuous Variables
# ===========
def bin_continuous(df, cols, bins=5, strategy='quantile'):
    """
    Discretize continuous variables into bins.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    cols : list
        Continuous columns to bin.
    bins : int, default=5
        Number of bins.
    strategy : {'quantile', 'uniform'}, default='quantile'
        Binning strategy.

    Returns
    -------
    pd.DataFrame
        Dataset with new '_bin' columns representing bins.
    """
    df_binned = df.copy()
    for col in cols:
        if strategy == 'quantile':
            df_binned[col + '_bin'] = pd.qcut(
                df_binned[col], q=bins, labels=False, duplicates='drop'
            )
        elif strategy == 'uniform':
            df_binned[col + '_bin'] = pd.cut(
                df_binned[col], bins=bins, labels=False
            )
    return df_binned


# ===========
# Function: Create Behavioral Flags (including smoker and non-smoker interaction flags)
# ===========
def create_behavior_flags(df):
    """
    Create domain-specific binary flags for high-risk and behavioral groups,
    including smoker and non-smoker interaction flags.

    Adds these flags:
    - is_high_risk: smoker AND BMI > 35 AND age > 60
    - is_young_smoker: smoker AND age < 30
    - is_high_risk_non_smoker: NON-smoker AND BMI > 35 AND age > 60
    - is_young_non_smoker: NON-smoker AND age < 30
    - age_nonsmoker: age * (non-smoker indicator)
    - bmi_nonsmoker: bmi * (non-smoker indicator)
    """

    df_flags = df.copy()

    # Standardize 'smoker' column to numeric binary (1 = smoker, 0 = non-smoker)
    if df_flags['smoker'].dtype == object:
        df_flags['smoker'] = df_flags['smoker'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # Create non-smoker indicator (1 if non-smoker, else 0)
    #df_flags['non_smoker'] = (df_flags['smoker'] == 0).astype(int)

    # Flag for high-risk smoker: smoker with high BMI and older age
    df_flags['is_high_risk'] = (
        (df_flags['smoker'] == 1) &
        (df_flags['bmi'] > 35) &
        (df_flags['age'] > 60)
    ).astype(int)

    # Flag for young smokers
    df_flags['is_young_smoker'] = (
        (df_flags['smoker'] == 1) &
        (df_flags['age'] < 30)
    ).astype(int)


    # Flag for young non-smoker
    #df_flags['is_young_non_smoker'] = (
    #    (df_flags['smoker'] == 0) &
    #    (df_flags['age'] < 30)
    #).astype(int)

    # Drop temporary 'non_smoker' indicator if you want, or keep for modeling
    # df_flags.drop(columns=['non_smoker'], inplace=True)

    return df_flags


# ===========
# Function: Drop Low-Correlation Features
# ===========
def drop_low_correlation_features(df, target_attr, threshold=0.05, method='spearman'):
    """
    Remove features whose absolute correlation with the target is <= threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_attr : str
        Target variable name.
    threshold : float, default=0.05
        Threshold for absolute correlation below which features will be dropped.
    method : str, default='spearman'
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    pd.DataFrame
        Dataset with low-correlation features removed.
    """
    corr_matrix = df.corr(method=method)
    target_corr = corr_matrix[target_attr].drop(target_attr)
    low_corr_features = target_corr[abs(target_corr) <= threshold].index.tolist()
    df_filtered = df.drop(columns=low_corr_features)
    print(f"Dropped {len(low_corr_features)} feature(s) due to low correlation: {low_corr_features}")
    return df_filtered


# ===========
# Function: Combine Categorical Columns
# ===========
def combine_categories(df, cols, new_col_name=None):
    """
    Combines two or more categorical columns into a single concatenated string column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    cols : list
        List of columns to combine.
    new_col_name : str, optional
        Name of the new combined column. If None, generated automatically.

    Returns
    -------
    pd.DataFrame
        Dataset with combined categorical column added.
    """
    df_combined = df.copy()
    if not isinstance(cols, list) or len(cols) < 2:
        raise ValueError("Please provide at least two columns to combine.")
    if new_col_name is None:
        new_col_name = "_".join(cols) + "_strata"
    df_combined[new_col_name] = df_combined[cols].astype(str).agg("_".join, axis=1)
    return df_combined


# ===========
# Function: Map to Nearest Region (string-safe)
# ===========
def map_to_nearest_region(df, region_col='region', valid_regions=None, new_col=None):
    """
    Performs fuzzy mapping of a region column to the closest matching region name
    from a list of valid regions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    region_col : str, default='region'
        Column name holding region names/values to be mapped.
    valid_regions : list
        List of valid region strings to map against.
    new_col : str, optional
        Name of new column to store mapped values; overwrites original if None.

    Returns
    -------
    pd.DataFrame
        Dataset with corrected region mapping.
    """
    if valid_regions is None:
        raise ValueError("Valid region list required.")
    df_mapped = df.copy()

    def nearest_region(val):
        match = get_close_matches(str(val), valid_regions, n=1, cutoff=0.0)
        return match[0] if match else val

    corrected = df_mapped[region_col].apply(nearest_region)
    if new_col:
        df_mapped[new_col] = corrected
    else:
        df_mapped[region_col] = corrected
    return df_mapped


# ===========
# Function: Upsample Categories with Numeric Perturbation
# ===========
def upsample_categories(df, upsample_cols, categorical_cols, numeric_cols_to_perturb=None,
                        comb_col='comb', random_state=42):
    """
    Upsample each category combination (defined by `upsample_cols`) to match the
    size of the largest group. Numeric columns can be slightly perturbed to avoid
    exact duplicate rows which may cause modeling issues.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    upsample_cols : list
        Categorical columns to combine into strata for upsampling.
    categorical_cols : list
        Columns to preserve as categorical dtypes.
    numeric_cols_to_perturb : list, optional
        Numeric columns on which to apply random small perturbations (default: None).
    comb_col : str, default='comb'
        Temporary column name for combined categories.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataset after upsampling and numeric perturbation, with temporary columns dropped.
    """
    # Combine categorical columns into a strata string for grouping
    df = combine_categories(df, upsample_cols, new_col_name=comb_col)
    grouped = df.groupby(comb_col)
    max_size = grouped.size().max()
    upsampled = []

    for _, group in grouped:
        # Upsample with replacement to max group size
        sampled = group.sample(n=max_size, replace=True, random_state=random_state).copy()

        # Apply small numeric perturbation to numeric columns if specified
        if numeric_cols_to_perturb:
            for col in numeric_cols_to_perturb:
                if col in sampled.columns and col not in categorical_cols:
                    # Slightly perturb values by +/- 0.5%
                    perturb_factors = np.random.uniform(0.995, 1.005, size=len(sampled))
                    sampled[col] = sampled[col] * perturb_factors

        # Ensure categorical dtypes are preserved
        for col in categorical_cols:
            if col in sampled.columns:
                sampled[col] = sampled[col].astype(df[col].dtype)

        upsampled.append(sampled)

    df_upsampled = pd.concat(upsampled).reset_index(drop=True)

    # Drop temporary combined strata column
    return df_upsampled.drop(columns=[comb_col])


# ===========
# Function: Full Fairness-Aware Data Transformation Pipeline
# ===========
def transform_data(df):
    """
    Applies a sequence of fairness-aware data transformation steps:

    1️⃣ Bin continuous variables (age, bmi) into quantile-based categories  
    2️⃣ Generate pairwise interaction features among age, bmi, smoker  
    3️⃣ Create behavioral flags including smoker and non-smoker flags and their interaction terms  
    4️⃣ Upsample strata defined by smoker status, age_bin, bmi_bin for fairness  
        while applying small numeric perturbations to avoid duplicates

    Returns:
        Transformed DataFrame ready for fairness-aware modeling.
    """
    print('1️⃣ Binning age and BMI...')
    df = bin_continuous(df, ['age', 'bmi'], bins=5, strategy='quantile')

    print('2️⃣ Creating interaction features...')
    #df['nonsmoker'] = 1 - df.smoker
    df = generate_interactions(df, ['age', 'bmi', 'smoker'], max_order=2)
    #df = generate_interactions(df, ['age', 'bmi', 'nonsmoker'], max_order=2)
    print('3️⃣ Adding behavioral flags...')
    df = create_behavior_flags(df)

    #print('4️⃣ Upsampling strata for fairness...')
    # Define categorical columns to keep their dtype
    categorical_cols = ['smoker', 'bmi_bin', 'age_bin', 'children', 'sex',
                        'is_high_risk', 'is_young_smoker', 'is_high_risk_non_smoker', 'is_young_non_smoker',
                        'age_nonsmoker', 'bmi_nonsmoker'] + \
                       [c for c in df.columns if c.startswith('region_')]

    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    df = upsample_categories(
        df,
        upsample_cols=['smoker', 'bmi_bin', 'age_bin'],
        categorical_cols=categorical_cols,
        numeric_cols_to_perturb=numeric_cols
    )

    return df
