# =====================================================
# 📦 Import Standard & Custom Libraries
# -----------------------------------------------------
# Libraries are grouped by functionality:
# - data handling (numpy, pandas)
# - visualization (matplotlib, seaborn)
# - modeling (xgboost, sklearn metrics)
# - presentation (IPython display for rich markdown)
# - custom project modules (EDA, fairness, explainability)
# =====================================================

## Import essential libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Rich output (for displaying markdown stages in reports)
from IPython.display import Markdown, display

# Base Model
from xgboost import XGBRegressor

# Performance Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Custom project modules (separate .py files in repo)
from explore import *            # EDA and visualization
from FairnessAnalysis import *   # Fairness metric computation
from ModelTrainer import *       # Training baseline vs fairness-constrained models
from SHAPAnalyzer import *       # Explainability via SHAP
from FairnessAuditor import *    # Group fairness & correction pipeline


# =====================================================
# Function: run_stage_pipeline
# Description:
#     This function represents the end-to-end execution of a single
#     modeling stage in the pipeline. It integrates:
#     1. Exploratory Data Analysis (EDA)
#     2. Fairness audits
#     3. Model training (baseline + fairness constrained)
#     4. Explainability analysis using SHAP
#     5. Fairness audit & correction (dual correction)
#
# Parameters:
#     model        : ML model object (e.g., XGBRegressor)
#     df           : Input dataset (pandas DataFrame)
#     target       : Name of target column (string)
#     hue          : Primary sensitive feature for fairness audit (string)
#     key_features : List of sensitive/group features to check fairness
#     stratify_by  : Features to stratify train-test split for fairness
#     metrics      : Dict of metrics {"MAE": func, "MSE": func, "R2": func}
#
# Returns:
#     List containing:
#     [data_fairness, model_results, SHAP_results, fairness_summary_report]
# =====================================================



import numpy as np
import pandas as pd

def get_fixed_size_sample_with_augmentation(
    df, size, noise_level=0.01, unique_threshold=10, random_state=42):
    """
    Samples exactly `size` records from the dataframe df.
    - Downsamples without replacement if df is larger than size.
    - Upsamples with replacement and applies ±noise_level augmentation to numeric columns if smaller.
    - Numeric columns defined as those with > unique_threshold unique values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    size : int
        Desired output sample size.
    noise_level : float, default 0.01
        Fractional noise range to apply to numeric columns when upsampling.
    unique_threshold : int, default 10
        Minimum unique values required to consider a column numeric.
    random_state : int, default 42
        Seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Sampled dataset of size `size`.
    """
    np.random.seed(random_state)
    n = len(df)
    numeric_cols = [col for col in df.columns 
                    if df[col].nunique() > unique_threshold and pd.api.types.is_numeric_dtype(df[col])]
    
    if n == size:
        return df.copy()
    elif n > size:
        return df.sample(n=size, random_state=random_state, replace=False).reset_index(drop=True)
    else:
        # Upsample with replacement
        df_upsampled = df.sample(n=size, random_state=random_state, replace=True).reset_index(drop=True)
        for col in numeric_cols:
            noise = np.random.uniform(1 - noise_level, 1 + noise_level, size)
            df_upsampled[col] = df_upsampled[col] * noise
        return df_upsampled

def run_stage_pipeline(model, df, target, hue, key_features, stratify_by, metrics, 
                       limit_records=int(1e4), noise_level=0.01, unique_threshold=10):
    """
    Runs the full ML pipeline:
    - Upsampling or downsampling the input data first to ensure consistent size.
    - Exploratory Data Analysis.
    - Data Fairness Audit.
    - Model Training.
    - SHAP explainability (on sampled data).
    - Model fairness audit and enhancement.
    
    Parameters
    ----------
    model : estimator
        ML model to train.
    df : pd.DataFrame
        Full dataset.
    target : str
        Target column name.
    hue, key_features, stratify_by : various
        Columns for analysis and fairness checks.
    metrics : dict
        Performance metrics dictionary for auditting.
    limit_records : int, default 10,000
        Number of records to sample for consistent training.
    noise_level : float, default 0.01
        Numeric augmentation noise fraction on upsampling.
    unique_threshold : int, default 10
        Threshold for defining numeric columns by unique values count.
    
    Returns
    -------
    List
        Results including fairness audits, model outputs, SHAP results etc.
    """
    
    # Step 0: Sample dataset to fixed size with augmentation if needed
    df_sampled = get_fixed_size_sample_with_augmentation(
        df, size=limit_records, noise_level=noise_level, unique_threshold=unique_threshold)
    print(f"[run_stage_pipeline] Input data resized to {len(df_sampled)} records.")
    
    # Step 1: Exploratory Data Analysis
    display(Markdown("## A. Exploratory Data Analysis"))
    eda(df, target, hue)
    
    # Step 2: Data Fairness Audit
    display(Markdown("## B. Data Fairness Audit"))
    data_fairness = Dataset_Fairness_Evaluation_Median(df=df, target=target, fairness_tags=key_features)
    display(data_fairness)
    
    # Step 3: Model Training
    display(Markdown("## C. Model Training"))
    trainer = ModelTrainer(df=df, model=model, target=target, stratify_by=stratify_by)
    model_results = trainer.run_fairness_comparison_pipeline()
    
    trained_baseline_model = model_results['baseline_model']
    trained_fair_model = model_results['fair_model']
    trained_subgroup_model = model_results['subgrp_model']
    
    # Step 4: SHAP explainability on sampled data
    display(Markdown("## D. SHAP Model explainability analysis"))
    analyzer = SHAPAnalyzer(model=trained_baseline_model, target_col=target, stratify_cols=stratify_by)
    SHAP_results = analyzer.run_analysis(data=df_sampled, top_interactions_n=15)
    
    # Step 5: Model Fairness Audits & Enhancement
    auditor = FairnessAuditor(df, target_col=target, sensitive_tags=key_features, 
                              model=trained_baseline_model, metrics_dict=metrics, stratify_col=stratify_by)
    fairness_summary_report_baseline = auditor.run_full_audit()
    
    auditor2 = FairnessAuditor(df, target_col=target, sensitive_tags=key_features, 
                               model=trained_subgroup_model, metrics_dict=metrics, stratify_col=stratify_by)
    fairness_summary_report_subgroup = auditor2.run_full_audit()
    
    # Return all relevant results
    return [data_fairness, model_results, SHAP_results, 
            [fairness_summary_report_baseline, fairness_summary_report_subgroup]]




def define_xgb():
    ## Revised model with reduced regularization
    xgb = XGBRegressor(
        n_estimators=250,
        objective='reg:squarederror',
        random_state=42,
        
        # Regularization (reduced)
        reg_alpha=0.1,        # Low L1 regularization
        reg_lambda=0.1,       # Low L2 regularization
    
        # Tree complexity (relaxed slightly)
        max_depth=20,         # Allow deeper trees
        min_child_weight=2,   # Allow splits with fewer samples
    
        # Subsampling (unchanged for now)
        subsample=0.8,
        colsample_bytree=0.8,
    
        # Learning rate (slightly increased for faster convergence)
        learning_rate=0.05
    )
    return xgb
