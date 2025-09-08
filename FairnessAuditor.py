"""
===========================================================
Fairness Auditor Class for Model Bias Detection and Mitigation
===========================================================

This module provides a comprehensive `FairnessAuditor` class to:
- Perform stratified data splitting for consistent evaluation.
- Generate predictions from pre-trained models.
- Apply post-processing fairness corrections using `FairnessByTrippleCorrection`.
- Compute overall and group-wise performance metrics (MAE, MSE, R2).
- Produce detailed comparative reports of model performance and fairness before and after correction.
- Visualize fairness impacts through plotting functions.

Designed Use-Cases:
-------------------
- Auditing the fairness of any pre-trained regression model across specified sensitive attributes.
- Quantifying the impact of a post-processing dual correction algorithm.
- Providing granular insights into performance disparities per group and feature.
- Supporting transparent and interpretable fairness evaluations for stakeholders.

Dependencies:
-------------
    numpy, pandas, sklearn, dual_correction (custom library), fairness_utils (custom library)

Author:
-------
    Ketan Bhatt
Date:
-----
    August 2025
===========================================================
"""

# Core Libraries
import numpy as np
import pandas as pd
import pickle
import warnings

# Modeling & Metrics from sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fairness & Correction: Custom Dual Correction Library
from dual_correction import FairnessByTrippleCorrection

# Imports from your utility files
# Assumed to be available in the environment or defined elsewhere
from fairness_utils import compute_multi_feature_fairness_df, plot_before_after_fairness_grid, compare_groupwise_influence
from train_model import get_stratified_split # Only get_stratified_split is needed from train_model.py

# Suppress specific warnings for cleaner output during execution
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with eval is deprecated")


class FairnessAuditor:
    """
    A comprehensive class to audit and apply dual correction to pre-trained predictive models,
    providing detailed insights into fairness and performance impacts.

    This class serves as a central hub for evaluating how well a model performs
    across different sensitive groups, and how post-processing bias mitigation
    techniques alter these outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset (features + target) used for evaluation.
    target_col : str
        The name of the target variable column in `df`.
    sensitive_tags : list of str
        A list of column names in `df` considered as sensitive attributes for fairness analysis.
    model : estimator
        A pre-trained, scikit-learn compatible regression model (must have a `.predict()` method).
        This model will be audited, not re-trained, by this class.
    metrics_dict : dict
        A dictionary of evaluation metrics (e.g., {"MAE": mean_absolute_error, "MSE": mean_squared_error}).
    stratify_col : str or list of str, optional
        Column name(s) to use for stratified train-test splitting within the auditor.
        Ensures proportional representation of these groups in splits for consistent auditing.
        If None, a simple ShuffleSplit will be used by `get_stratified_split`.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split for auditing.
    random_state : int, default=42
        Controls the randomness of data splitting for reproducibility.

    Attributes
    ----------
    X_train, y_train, X_test, y_test : pd.DataFrame, pd.Series
        Data splits generated internally by the auditor.
    uncorrected_preds : np.ndarray
        Predictions from the `self.model` on the test set *before* dual correction.
    corrected_preds : np.ndarray
        Predictions on the test set *after* applying dual correction.
    before_fairness_df, after_fairness_df : pd.DataFrame
        DataFrames containing group-wise fairness metrics before and after correction.
    performance_comparison_df : pd.DataFrame
        DataFrame summarizing overall model performance metrics (e.g., RMSE, MAE, R2)
        before and after dual correction.
    comparison_df : pd.DataFrame
        DataFrame detailing group-wise influence comparison before and after correction.
    fairness_summary_df : pd.DataFrame
        Consolidated summary report of fairness improvement, including mean and standard deviation changes.
    dc_model : FairnessByTrippleCorrection
        The fitted dual correction model.
    audit_md : dict
        Dictionary containing markdown-formatted audit logs from `FairnessByTrippleCorrection`.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, sensitive_tags: list[str], model, metrics_dict: dict,
                 stratify_col: str | list[str] | None = None, test_size: float = 0.2, random_state: int = 42):
        self.df = df
        self.target_col = target_col
        self.sensitive_tags = sensitive_tags
        self.model = model  # This is the pre-trained model to be audited
        self.metrics_dict = metrics_dict
        self.stratify_col = stratify_col
        self.test_size = test_size
        self.random_state = random_state

        # Initialize attributes to store results
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.corrected_preds = None
        self.uncorrected_preds = None
        self.before_fairness_df = None
        self.after_fairness_df = None
        self.comparison_df = None
        self.performance_comparison_df = None
        self.fairness_summary_df = None
        self.dc_model = None
        self.audit_md = {}

    def stratified_split(self):
        """
        Performs a stratified train-test split on the input dataset (`self.df`).
        This method ensures that the proportions of the specified `stratify_col`
        are maintained in both the training and testing sets, which is crucial
        for consistent and reliable fairness auditing across subgroups.

        The split dataframes are stored as attributes of the auditor instance.

        Raises
        ------
        ValueError
            If `self.df` is empty or if `get_stratified_split` encounters issues.
        """
        print(f"- Splitting data (stratified by {self.stratify_col})...")
        # get_stratified_split is assumed to handle the logic for single or list stratify_by
        self.X_train, self.y_train, self.X_test, self.y_test = get_stratified_split(
            self.df, target=self.target_col, stratify_by=self.stratify_col,
            test_size=self.test_size, random_state=self.random_state
        )
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Data splitting failed or resulted in empty sets.")
        print("  - Data split done.")


    def generate_predictions_and_scores(self):
        """
        Generates predictions using the `self.model` (which is assumed to be
        pre-trained) on the auditor's internal `X_train` and `X_test` data.
        It then computes overall performance scores for these uncorrected predictions.

        This method does NOT re-train `self.model`. It uses the model as provided
        in the constructor.

        Returns
        -------
        tuple
            - scores (dict): Dictionary of overall performance metrics (RMSE, MAE, R2)
              for both train and test sets using the uncorrected predictions.
            - y_pred_train (np.ndarray): Predictions on the training set.
            - y_pred_test (np.ndarray): Predictions on the test set (uncorrected).

        Raises
        ------
        RuntimeError
            If `self.model` has not been fitted or cannot predict.
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data splits not available. Call stratified_split() first.")

        print("- Generating uncorrected predictions...")
        # Assume self.model is already fitted; just predict
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        self.uncorrected_preds = y_pred_test # Store for later use

        # Calculate comprehensive overall scores for these uncorrected predictions
        # These will be the 'Before DC' scores in the overall performance table.
        scores = {
            'rmse_train': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'mae_train': mean_absolute_error(self.y_train, y_pred_train),
            'r2_train': r2_score(self.y_train, y_pred_train),
            'rmse_test': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'mae_test': mean_absolute_error(self.y_test, y_pred_test),
            'r2_test': r2_score(self.y_test, y_pred_test)
        }
        print("  - Uncorrected predictions and scores calculated.")
        return scores, y_pred_train, y_pred_test

    def apply_dual_correction(self, y_pred_train: np.ndarray, y_pred_test: np.ndarray):
        """
        Applies dual fairness correction to the model predictions (`y_pred_test`).

        This method fits the `FairnessByTrippleCorrection` model using the training data
        and predictions, and then applies the learned corrections to the test set predictions.
        It also generates an audit markdown for immediate group-wise comparison.

        Parameters
        ----------
        y_pred_train : np.ndarray
            Raw (uncorrected) model predictions for the training set.
        y_pred_test : np.ndarray
            Raw (uncorrected) model predictions for the test set.

        Returns
        -------
        tuple
            - corrected_preds (np.ndarray): Test predictions after applying fairness corrections.
            - audit_md (dict): Per-feature audit logs showing before & after group-level means.

        Side Effects
        ------------
        - Stores the corrected predictions in `self.corrected_preds`.
        - Stores the audit markdown in `self.audit_md`.
        - Stores the fitted `FairnessByTrippleCorrection` model in `self.dc_model`.
        - Saves the `dc_model` to "fairness_dual_corrector.pkl".
        """
        print("- Splitting train/test datasets for DC application...")
        # 1️⃣ Prepare train set dataframe including sensitive features, target, and predictions
        df_train_dc = self.X_train[self.sensitive_tags].copy()
        df_train_dc['y_true'] = self.y_train
        df_train_dc['y_pred'] = y_pred_train
        df_train_dc['split'] = 'train'

        # 2️⃣ Prepare test set dataframe
        df_test_dc = self.X_test[self.sensitive_tags].copy()
        df_test_dc['y_true'] = self.y_test
        df_test_dc['y_pred'] = y_pred_test
        df_test_dc['split'] = 'test'
        print("  - Dataframes prepared for DC.")

        # 3️⃣ Combine train and test for fitting the dual corrector
        df_combined_dc = pd.concat([df_train_dc, df_test_dc], axis=0)
        print("  - Combined train and test data for DC fitting.")

        # 4️⃣ Initialize FairnessByTrippleCorrection object
        # You can add configurable parameters here if needed, e.g., apply_global, objective etc.
        #dc = FairnessByDualCorrection()
        dc = FairnessByTrippleCorrection()

        print("- Calculating global shift...")
        print("- Calculating group-wise offsets...")
        # 5️⃣ Fit correction model on combined data to learn shifts/offsets
        # The `fit_from_dataframe` method handles identifying sensitive groups and calculating offsets.
        # It returns `group_test` which is the sensitive features for the test set, aligned.
        _, group_test_aligned = dc.fit_from_dataframe(
            df=df_combined_dc,
            target_col='y_true',  # True values for training the corrector
            pred_col='y_pred',    # Predictions for training the corrector
            sensitive_cols=self.sensitive_tags, # Sensitive features to consider for correction
            split_col='split',
            train_tag='train'
        )
        print("  - Global shift and group-wise offsets calculated.")

        # 6️⃣ Apply correction with before/after audit
        # `transform_with_audit` applies the learned corrections and returns corrected predictions
        # along with a detailed audit (before vs after group means).
        #corrected_preds, audit_md = dc.transform_with_audit(
        #    y_pred_test,         # BEFORE predictions for the test set
        #    group_test_aligned    # Sensitive group tags aligned with the test set
        #)

        # --- CHANGE TO THE FOLLOWING ---
        # Explicitly convert NumPy arrays to Pandas Series, retaining the original index
        y_pred_test_series = pd.Series(y_pred_test, index=self.X_test.index)
        group_test_aligned_series = pd.Series(group_test_aligned, index=self.X_test.index)
        
        # Now pass these Series objects to the dual correction's transform_with_audit method
        corrected_preds, audit_md = dc.transform_with_audit(
            y_pred_test_series,         # BEFORE predictions for the test set, now as Series
            group_test_aligned_series   # Sensitive group tags aligned with the test set, now as Series
        )   
        
        print("  - Dual correction applied to test predictions.")
        
        # 7️⃣ Save trained corrector for reuse and store in instance
        try:
            with open("fairness_dual_corrector.pkl", "wb") as f:
                pickle.dump(dc, f)
            print("  - Dual correction model saved to 'fairness_dual_corrector.pkl'.")
        except Exception as e:
            print(f"⚠️ Warning: Could not save fairness_dual_corrector.pkl: {e}")

        self.dc_model = dc
        self.corrected_preds = corrected_preds
        self.audit_md = audit_md
        print("  - Corrected predictions and audit data stored in auditor instance.")

    def compute_fairness_metrics(self):
        """
        Computes detailed group-wise fairness metrics for both the uncorrected
        and dual-corrected predictions. These metrics include MAE, MSE, and R2,
        broken down by each specified sensitive feature.

        It also prints the raw audit markdown provided by `FairnessByTrippleCorrection`.

        Returns
        -------
        tuple
            - before_fairness_df (pd.DataFrame): Group-wise metrics before correction.
            - after_fairness_df (pd.DataFrame): Group-wise metrics after correction.
        """
        print("📋 Computing group-wise fairness metrics...")

        # Display raw audit markdown from DualCorrection for immediate feedback
        for s_tag in self.sensitive_tags:
            print(f"\n### Audit for `{s_tag}` ###\n")
            print(self.audit_md.get(s_tag, "⚠️ No audit markdown available for this sensitive feature."))

        # Compute comprehensive fairness DataFrames using utility function
        # This will calculate MAE, MSE, R2 for each subgroup
        self.before_fairness_df = compute_multi_feature_fairness_df(
            y_true=self.y_test,
            y_pred=self.uncorrected_preds,
            df_features=self.X_test[self.sensitive_tags], # Pass only the sensitive features for grouping
            sensitive_features=self.sensitive_tags,
            metrics_dict=self.metrics_dict
        )
        self.after_fairness_df = compute_multi_feature_fairness_df(
            y_true=self.y_test,
            y_pred=self.corrected_preds,
            df_features=self.X_test[self.sensitive_tags], # Pass only the sensitive features for grouping
            sensitive_features=self.sensitive_tags,
            metrics_dict=self.metrics_dict
        )
        print("  - Group-wise fairness metrics computed.")
        return self.before_fairness_df, self.after_fairness_df

    def _create_performance_comparison_table(self, before_scores: dict) -> pd.DataFrame:
        """
        Calculates post-correction overall performance metrics and creates a comparison DataFrame.
        This method computes overall performance metrics (RMSE, MAE, R2) for the model
        *before* and *after* any dual correction (applied externally).

        Parameters
        ----------
        before_scores : dict
            A dictionary containing the 'before dual correction' overall performance scores
            (e.g., 'rmse_train', 'mae_train', 'r2_train', 'rmse_test', 'mae_test', 'r2_test').
            These should come directly from the evaluation of the model passed to the auditor.

        Returns
        -------
        pd.DataFrame
            A DataFrame comparing the overall performance metrics before and after dual correction.
        """
        # Calculate post-correction test scores from `self.corrected_preds`
        after_scores_test = {
            'rmse_test': np.sqrt(mean_squared_error(self.y_test, self.corrected_preds)),
            'mae_test': mean_absolute_error(self.y_test, self.corrected_preds),
            'r2_test': r2_score(self.y_test, self.corrected_preds)
        }

        # Combine training scores (which are 'before' scores, as dual correction is post-processing)
        # with the 'after' test scores. Training performance remains unchanged by post-processing.
        after_scores = {
            'rmse_train': before_scores['rmse_train'],
            'mae_train': before_scores['mae_train'],
            'r2_train': before_scores['r2_train'],
            **after_scores_test
        }

        # Create DataFrame for clear comparison
        comparison_df = pd.DataFrame({"Before DC": before_scores, "After DC": after_scores})
        comparison_df['Impact'] = comparison_df['After DC'] - comparison_df['Before DC']
        comparison_df = comparison_df.reset_index().rename(columns={'index': 'Parameter'})
        return comparison_df

    def summarize_fairness_improvement_report(self, fairness_report_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarizes feature-wise and overall fairness improvements with simplified statistics.

        This function aggregates the detailed group-wise metrics (MAE, MSE, R2)
        from `fairness_report_df` to provide a concise summary for each feature.
        It calculates mean and standard deviation of metric values across groups
        before and after correction, along with their changes.

        Parameters
        ----------
        fairness_report_df : pd.DataFrame
            A DataFrame containing group-wise fairness metrics with columns
            ['Feature', 'Group', 'Metric', 'Value_before', 'Value_after'].
            This DataFrame is typically generated by `compare_groupwise_influence`.

        Returns
        -------
        pd.DataFrame
            A summary table with columns:
            'Feature', 'Metric', 'Mean_before', 'Stddev_before',
            'Mean_after', 'Stddev_after', 'ΔMean', 'ΔStddev'.
            Includes an "Overall {Metric}" row for each metric aggregated across all features.
        """
        metrics = fairness_report_df['Metric'].unique()
        features = fairness_report_df['Feature'].unique()

        summary_rows = []

        for metric in metrics:
            df_metric = fairness_report_df[fairness_report_df['Metric'] == metric]

            for feature in features:
                df_feat = df_metric[df_metric['Feature'] == feature]

                if df_feat.empty:
                    continue

                mean_before = np.mean(df_feat['Value_before'])
                std_before = np.std(df_feat['Value_before'])
                mean_after = np.mean(df_feat['Value_after'])
                std_after = np.std(df_feat['Value_after'])

                delta_mean = mean_after - mean_before
                delta_std = std_after - std_before

                summary_rows.append({
                    'Feature': feature,
                    'Metric': metric,
                    'Mean_before': mean_before,
                    'Stddev_before': std_before,
                    'Mean_after': mean_after,
                    'Stddev_after': std_after,
                    'ΔMean': delta_mean,
                    'ΔStddev': delta_std
                })

            # Add an "Overall aggregated" row per metric, aggregating across all features
            mean_before_overall = np.mean(df_metric['Value_before'])
            std_before_overall = np.std(df_metric['Value_before'])
            mean_after_overall = np.mean(df_metric['Value_after'])
            std_after_overall = np.std(df_metric['Value_after'])

            summary_rows.append({
                'Feature': f'Overall {metric}',
                'Metric': metric,
                'Mean_before': mean_before_overall,
                'Stddev_before': std_before_overall,
                'Mean_after': mean_after_overall,
                'Stddev_after': std_after_overall,
                'ΔMean': mean_after_overall - mean_before_overall,
                'ΔStddev': std_after_overall - std_before_overall
            })

        summary_df = pd.DataFrame(summary_rows)
        # Sort for consistent display in reports
        return summary_df.sort_values(['Metric', 'Feature']).reset_index(drop=True)

    def run_full_audit(self):
        """
        Executes the complete fairness audit pipeline from data splitting,
        prediction generation from the provided model, fairness correction,
        to comprehensive reporting and visualization.

        This method orchestrates all steps to generate a full fairness audit report
        for the `self.model` (the pre-trained model passed to the auditor).

        Returns
        -------
        pd.DataFrame
            The summary DataFrame (`self.fairness_summary_df`) comparing
            group-wise influence before and after fairness correction.
        """
        # Ensure display is available in the current environment
        from IPython.display import display

        print("--- Starting Fairness Audit Pipeline ---")

        # Step 1: Perform stratified train-test split for consistent auditing
        # This split is performed anew for each auditor instance to ensure consistent evaluation context.
        print("\n🔧 Step 1: Stratified split...")
        self.stratified_split()

        # Step 2: Generate predictions from the provided pre-trained model and evaluate overall scores
        # This `scores` object will be the "Before DC" reference for overall performance.
        print("\n📊 Step 2: Generating predictions and evaluating overall scores (Before DC)...")
        scores, y_pred_train, y_pred_test = self.generate_predictions_and_scores()

        # Step 3: Apply fairness correction method(s) (Dual Correction)
        print("\n📈 Step 3: Applying dual correction...")
        self.apply_dual_correction(y_pred_train, y_pred_test)

        # Step 4: Create and display overall performance comparison table (Before vs After DC)
        print("\n🔍 Step 4: Comparing overall model performance...")
        self.performance_comparison_df = self._create_performance_comparison_table(scores)
        display(self.performance_comparison_df)
        print("  - Overall performance comparison done.")
            
        # Step 5: Compute and display group-wise fairness evaluation metrics
        # This populates self.before_fairness_df and self.after_fairness_df
        print("\n📋 Step 5: Computing group-wise fairness metrics...")
        self.compute_fairness_metrics()
        print("  - Group-wise fairness metrics computed.")

        # Step 6: Plot fairness grids for corrected predictions visualization
        print("\n🎨 Step 6: Plotting fairness grid for corrected predictions...")
        # `plot_before_after_fairness_grid` is assumed to be implemented in fairness_utils.py
        plot_before_after_fairness_grid(self.before_fairness_df, self.after_fairness_df)
        print("  - Fairness grids plotted.")

        # Step 7: Compare group-wise influence before and after correction
        # This creates the detailed comparison DataFrame.
        print("\n⚖️ Step 7: Comparing group-wise influence (detailed breakdown)...")
        # `compare_groupwise_influence` is assumed to be implemented in fairness_utils.py
        self.comparison_df = compare_groupwise_influence(
            self.before_fairness_df, self.after_fairness_df
        )
        print("  - Group-wise influence comparison done.")

        # Step 8: Summarize feature-wise and overall fairness improvements
        print("\n📝 Step 8: Summarizing fairness improvement report...")
        if self.comparison_df is not None:
            self.fairness_summary_df = self.summarize_fairness_improvement_report(
                self.comparison_df
            )
            # Display summary tables for each metric (MAE, MSE, R2)
            for m in ['MAE', 'MSE', 'R2']:
                print(f'\n\nFor Metric: {m}\n')
                display(self.fairness_summary_df[self.fairness_summary_df.Metric == m])
        else:
            print("⚠️ Fairness improvement report not available; skipping summary.")
        print("  - Fairness improvement report summarized.")

        print("\n✅ Fairness audit complete.")
        return self.fairness_summary_df

