"""
===========================================================
Model Training, Evaluation, and Fairness-Aware Pipeline
===========================================================

This module provides utilities to:
- Train, evaluate, and compare baseline and fairness-aware regression models.
- Apply fairness constraints using `fairlearn` reductions (Exponentiated Gradient + BoundedGroupLoss).
- Compute standard and per-group performance metrics.
- Generate visualizations: predictions vs actual values, and feature importance.

Designed Use-Cases:
-------------------
- Evaluate performance of regression models in contexts where fairness across subgroups matters.
- Conduct bias/fairness audits by comparing baseline model behavior with fairness-constrained models.
- Visualize the impact of fairness constraints on prediction accuracy and fairness metrics.

Techniques Employed:
--------------------
- Stratified data splitting (to maintain subgroup proportions of the `stratify_by` feature(s)).
- RMSE, MAE, R² evaluation metrics for overall and per-group performance.
- In-processing fairness algorithm (Exponentiated Gradient).
- Group loss constraints for error parity using `BoundedGroupLoss`.
- Hybrid subgroup modeling using a dedicated model for a key sensitive attribute.

Dependencies:
-------------
    pandas, numpy, sklearn, fairlearn, matplotlib, IPython

Author:
-------
    Your AI Assistant
Date:
-----
    August 2025
===========================================================
"""

# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import display, Markdown

# Fairlearn specific imports
from fairlearn.reductions import ExponentiatedGradient, BoundedGroupLoss

class SKlearnMetricWrapper:
    """
    Adapter to make sklearn-compatible metrics usable inside
    fairlearn's `BoundedGroupLoss` constraint framework.

    This wrapper exposes a `.eval()` method which conforms
    to fairlearn's expectations for a loss function.
    """

    def __init__(self, metric_fn: callable):
        """
        Parameters
        ----------
        metric_fn : callable
            A metric function following sklearn's (y_true, y_pred) signature.
            Example: sklearn.metrics.mean_absolute_error.
        """
        self.metric_fn = metric_fn

    def eval(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate wrapped metric.

        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted target values.

        Returns
        -------
        float
            The scalar metric value.
        """
        return self.metric_fn(y_true, y_pred)


"""
===========================================================
ModelTrainer Class for Stratified Regression with Fairness Constraints
===========================================================

This class provides a full pipeline to:
- Perform stratified train/test splits (including by multiple columns)
- Train and evaluate baseline regression models
- Train fairness-aware models using Fairlearn's ExponentiatedGradient with BoundedGroupLoss constraints
- Compare overall and per-group performance metrics for baseline vs fair models
- Visualize predictions and feature importances for both models

Dependencies:
-------------
pandas, numpy, sklearn, fairlearn, matplotlib, IPython.display

Author:
-------
Your AI Assistant
Date: August 2025
===========================================================
"""


class ModelTrainer:
    """
    Class encapsulating a regression model training and fairness comparison pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including features and target.
    model : estimator
        A scikit-learn compatible regression model.
    target : str, default='charges'
        Name of the target variable column.
    stratify_by : str or list of str, default='smoker'
        Column name(s) to use for stratified train-test splitting.
        Ensures proportional representation of these groups in splits.
    sensitive_features : str or list of str, optional
        Column name(s) to consider as sensitive attributes for fairness evaluation
        and constraint application. If None, defaults to `stratify_by`.
        These are the features for which fairness (e.g., error parity) is desired.
    test_size : float, default=0.2
        Proportion of dataset assigned to the test set.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, df: pd.DataFrame, model, target: str = 'charges',
                 stratify_by: str | list[str] = 'smoker',
                 sensitive_features: str | list[str] | None = None,
                 test_size: float = 0.2, random_state: int = 42):
        self.df = df.copy()
        self.model = model
        self.target = target
        self.stratify_by = stratify_by
        self.test_size = test_size
        self.random_state = random_state

        # Ensure sensitive_features is always a list for consistent handling
        # If not explicitly provided, use stratify_by for fairness as well.
        if sensitive_features is None:
            self.sensitive_features = [stratify_by] if isinstance(stratify_by, str) else stratify_by
        else:
            self.sensitive_features = [sensitive_features] if isinstance(sensitive_features, str) else sensitive_features

        # Initialize data containers to None
        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None
        self.y_pred_train: np.ndarray | None = None # Holds last trained model's train predictions
        self.y_pred_test: np.ndarray | None = None # Holds last trained model's test predictions
        self.scores: dict | None = None # Stores overall training scores (e.g., from baseline model for dynamic upper bound)
        self._sensitive_features_df_train: pd.DataFrame | None = None # Sensitive features for training Fairlearn model
        self._sensitive_features_df_test: pd.DataFrame | None = None # Sensitive features for testing Fairlearn model

    def stratified_split(self):
        """
        Perform a stratified train/test split on the dataset.

        Uses the `stratify_by` column(s) to maintain proportional representation
        of groups in both training and testing sets.
        Also extracts the specified `sensitive_features` for the training and test sets
        to be used by Fairlearn.
        """
        if self.X_train is not None:
            print("Data already split. Skipping stratified_split().")
            return

        df_processed = self.df.copy()

        # Create a combined key for stratification if stratify_by is a list,
        # otherwise use the single column. Convert to string to avoid issues
        # with non-hashable types in stratification.
        # This part ensures compliance for list-based 'stratify_by' for sklearn's splitter.
        if isinstance(self.stratify_by, list):
            # Combine multiple columns into a single string for stratification
            # This is similar to the user's provided `combine_categories` function.
            stratify_col = df_processed[self.stratify_by].astype(str).agg('_'.join, axis=1)
            print(f"✅ Stratifying by combined columns: {self.stratify_by}")
        else:
            stratify_col = df_processed[self.stratify_by].astype(str) # Ensure it's string for consistent hashing
            print(f"✅ Stratifying by single column: {self.stratify_by}")

        # Fallback to ShuffleSplit if stratification is not feasible
        # (e.g., too many unique groups, or groups with only one sample)
        # StratifiedShuffleSplit requires at least 2 samples for each class in both train/test sets.
        if stratify_col.nunique() > 50 or (stratify_col.value_counts() < 2).any():
            print(f"⚠️ Warning: Stratification column '{self.stratify_by}' has too many unique values "
                  f"({stratify_col.nunique()}) or groups with fewer than 2 samples. "
                  f"Proceeding with standard ShuffleSplit instead of StratifiedShuffleSplit.")
            splitter = ShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            # For ShuffleSplit, the `split` method only takes features, no `y` for stratification.
            split_iterator = splitter.split(df_processed.drop(columns=[self.target]))
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            split_iterator = splitter.split(df_processed.drop(columns=[self.target]), stratify_col)


        features = df_processed.drop(columns=[self.target])
        labels = df_processed[self.target]

        for train_idx, test_idx in split_iterator:
            self.X_train = features.iloc[train_idx].copy()
            self.y_train = labels.iloc[train_idx]
            self.X_test = features.iloc[test_idx].copy()
            self.y_test = labels.iloc[test_idx]

            # Extract sensitive features as DataFrames (or Series if single column) for Fairlearn.
            # Fairlearn expects this format and can handle multiple sensitive features here.
            self._sensitive_features_df_train = df_processed.iloc[train_idx][self.sensitive_features].copy()
            self._sensitive_features_df_test = df_processed.iloc[test_idx][self.sensitive_features].copy()
            break # Only need one split

    def train_model(self, model_to_train=None):
        """
        Fit the provided model (or default) on the training data.

        Parameters
        ----------
        model_to_train : estimator, optional
            Model to fit. If None, a clone of the original model provided during
            initialization is used.

        Returns
        -------
        fitted estimator
            The trained model instance.
        """
        if self.X_train is None:
            raise ValueError("Data not split. Call stratified_split() or run_pipeline() first.")

        if model_to_train is None:
            model_to_train = clone(self.model) # Clone to ensure original model instance isn't modified

        model_to_train.fit(self.X_train, self.y_train)
        return model_to_train

    def evaluate_model(self, trained_model, X_data: pd.DataFrame, y_true: pd.Series):
        """
        Evaluate the model on a given dataset using RMSE, MAE, and R².

        Parameters
        ----------
        trained_model : estimator
            Model to evaluate.
        X_data : pd.DataFrame
            Feature data to predict on.
        y_true : pd.Series
            True target values for evaluation

        Returns
        -------
        tuple[np.ndarray, dict]
            Tuple containing:
            - y_pred: Predicted values as a NumPy array.
            - scores: Dictionary of evaluation scores (RMSE, MAE, R²).
        """
        y_pred = trained_model.predict(X_data)

        scores = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
        # The corrected print statement
        print(f"RMSE: {scores['rmse']:.2f} MAE: {scores['mae']:.2f} R2: {scores['r2']:.2f}")
        return y_pred, scores

    def get_performance_df(self, train_scores: dict, test_scores: dict) -> pd.DataFrame:
        """
        Return overall evaluation scores as a DataFrame, indexed by train/test.

        Parameters
        ----------
        train_scores : dict
            Dictionary of scores for the training set.
        test_scores : dict
            Dictionary of scores for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame with RMSE, MAE, and R² scores for train & test.
        """
        return pd.DataFrame(
            {
                'RMSE': [train_scores["rmse"], test_scores["rmse"]],
                'MAE': [train_scores["mae"], test_scores["mae"]],
                'R²': [train_scores["r2"], test_scores["r2"]],
            },
            index=['Training', 'Test'],
        )

    def train_fair_model(self, upper_bound_val: float | None = None):
        """
        Train a fairness-aware model applying fairness constraints on the specified
        sensitive feature(s) using Fairlearn's ExponentiatedGradient.

        Uses BoundedGroupLoss on MAE as the constraint.

        Parameters
        ----------
        upper_bound_val : float, optional
            Upper bound on allowed group loss difference for fairness.
            If None, uses 1/3 of the baseline model's MAE on the training set.

        Returns
        -------
        fairlearn.reductions.ExponentiatedGradient
            The fitted fairness-aware model.
        """
        # Explicitly import Fairlearn components here for robustness.
        # This ensures they are always available when this method is called,
        # even if the outer script doesn't import them globally.
        from fairlearn.reductions import ExponentiatedGradient, BoundedGroupLoss

        if self.X_train is None:
            raise ValueError("Data not split. Call stratified_split() or run_fairness_comparison_pipeline() first.")
        if self._sensitive_features_df_train is None:
            raise ValueError("Sensitive features data not extracted. Ensure stratified_split() was called.")

        base_model = clone(self.model)

        # Define the upper bound dynamically if not provided
        if upper_bound_val is None:
            if self.scores is None:
                raise ValueError("Baseline model not evaluated. Cannot dynamically set upper_bound. "
                                 "Run the full comparison pipeline first to establish baseline scores.")
            # Use the MAE from the baseline model's training set as a reference
            baseline_train_mae = self.scores["mae"]
            upper_bound = baseline_train_mae / 3
            print(f"✅ Dynamic `upper_bound` set to 1/3 of baseline training MAE: {upper_bound:.2f}")
        else:
            upper_bound = upper_bound_val

        mae_wrapper = SKlearnMetricWrapper(mean_absolute_error)
        # BoundedGroupLoss tries to ensure that the error (MAE in this case) for each group
        # is within a certain bound of the minimum error observed across groups.
        constraint = BoundedGroupLoss(loss=mae_wrapper, upper_bound=upper_bound)

        # ExponentiatedGradient is an in-processing algorithm that trains a series of
        # reweighted base models to satisfy the fairness constraints.
        fair_model = ExponentiatedGradient(estimator=base_model, constraints=constraint)

        print(f"\n📈 Training fairness-constrained model (constraints on {self.sensitive_features})...")
        print("This process may take longer due to fairness optimization.")

        fair_model.fit(
            self.X_train,
            self.y_train,
            sensitive_features=self._sensitive_features_df_train, # Pass the sensitive features DataFrame
        )

        # Evaluate fairness constraint violations on training predictions
        try:
            # `constraint.gamma` calculates the constraint violations for a given predictor.
            # It expects a callable predictor, X_data, y_true, and sensitive_features.
            violations = constraint.gamma(
                lambda X: fair_model.predict(X), # A lambda function acting as the predictor
                self.X_train,
                self.y_train,
                sensitive_features=self._sensitive_features_df_train
            )
            print(f"Maximum group loss violation on training set for Fair Model: {violations.max():.4f}")
        except Exception as e:
            print(f"⚠️ Unable to compute constraint violations on training set for Fair Model: {e}. "
                  "This might happen with certain model types or if gamma calculation is complex. "
                  "Safe to ignore if primarily using overall performance and group-wise metrics.")

        return fair_model

    def compute_groupwise_metrics(self, y_true: pd.Series, y_pred: np.ndarray, sensitive_groups: pd.Series) -> pd.DataFrame:
        """
        Compute per-group MAE, MSE, R², and count metrics.

        Also calculates the mean and standard deviation of MAE, MSE, and R² across groups,
        providing a summary of disparity.

        Parameters
        ----------
        y_true : pd.Series
            True target values.
        y_pred : np.ndarray
            Predicted target values.
        sensitive_groups : pd.Series
            Grouping feature (e.g., 'smoker') for which to compute metrics per group.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'group', 'MAE', 'MSE', 'R2', 'Count',
            and additional rows for the mean and standard deviation of metrics across groups.
        """
        # Create a temporary DataFrame for easy grouping
        df_group_metrics = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive_groups})

        metrics = (
            df_group_metrics.groupby("group")
            .apply(
                lambda g: pd.Series(
                    {
                        "MAE": mean_absolute_error(g.y_true, g.y_pred),
                        "MSE": mean_squared_error(g.y_true, g.y_pred),
                        "R2": r2_score(g.y_true, g.y_pred),
                        "Count": len(g),
                    }
                )
            )
            .reset_index()
        )

        # Calculate mean and standard deviation of MAE, MSE, and R2 across groups
        # These provide a summary of the disparities.
        mean_metrics_row = pd.DataFrame([['Mean',
                                          np.mean(metrics['MAE']),
                                          np.mean(metrics['MSE']),
                                          np.mean(metrics['R2']),
                                          np.nan]], # Count doesn't have a meaningful mean here
                                        columns=['group', 'MAE', 'MSE', 'R2', 'Count'])
        std_metrics_row = pd.DataFrame([['StdDev',
                                         np.std(metrics['MAE']),
                                         np.std(metrics['MSE']),
                                         np.std(metrics['R2']),
                                         np.nan]], # Count doesn't have a meaningful std dev here
                                        columns=['group', 'MAE', 'MSE', 'R2', 'Count'])

        # Concatenate the group-wise metrics with the summary rows
        metrics = pd.concat([metrics, mean_metrics_row, std_metrics_row], ignore_index=True)

        return metrics

    def plot_predictions(self, y_train: pd.Series, y_pred_train: np.ndarray,
                         y_test: pd.Series, y_pred_test: np.ndarray, title_prefix: str = ""):
        """
        Plot predicted vs actual values scatterplots for train and test sets.

        Shows best-fit regression line and reference y=x line.

        Parameters
        ----------
        y_train : pd.Series
            True target values for the training set.
        y_pred_train : np.ndarray
            Predicted values for the training set.
        y_test : pd.Series
            True target values for the test set.
        y_pred_test : np.ndarray
            Predicted values for the test set.
        title_prefix : str, default=""
            Prefix for the plot titles (e.g., "Baseline " or "Fair ").
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        def plot_fit(ax, y_true, y_pred, title, color, line_color):
            """Helper function to plot scatter and regression line."""
            # Ensure y_true is 2D for sklearn's fit method (expected input format)
            fit = LinearRegression().fit(y_true.values.reshape(-1, 1), y_pred)
            line = fit.predict(y_true.values.reshape(-1, 1))
            slope = fit.coef_[0]
            intercept = fit.intercept_

            ax.scatter(y_true, y_pred, alpha=0.2, color=color)
            ax.plot(y_true, line, color=line_color, label="Best Fit")
            ref_line = np.linspace(y_true.min(), y_true.max(), 100)
            ax.plot(ref_line, ref_line, linestyle="--", color="gray", label="y = x")

            ax.set_title(title)
            ax.set_xlabel("Actual Charges")
            ax.set_ylabel("Predicted Charges") # Added Y-label for clarity
            ax.legend()
            return slope, intercept

        # Plot for training data
        s_train, i_train = plot_fit(
            axes[0], y_train, y_pred_train, f"{title_prefix}Training: Predictions vs Actual", "skyblue", "navy"
        )
        # Plot for test data
        s_test, i_test = plot_fit(
            axes[1], y_test, y_pred_test, f"{title_prefix}Test: Predictions vs Actual", "coral", "darkred"
        )
        print(f"\n📐 {title_prefix}Training Best-Fit Line: y = {s_train:.4f}x + {i_train:.2f}")
        print(f"\n📐 {title_prefix}Test Best-Fit Line: y = {s_test:.4f}x + {i_test:.2f}")

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model, X_data: pd.DataFrame, title_prefix: str = ""):
        """
        Plot horizontal bar chart of feature importance if available.

        Supports `.coef_` for linear models and `.feature_importances_` for tree-based models.
        For Fairlearn's ExponentiatedGradient, it attempts to average importances
        from its underlying base estimators.

        Parameters
        ----------
        model : estimator
            The trained model (can be a scikit-learn model or Fairlearn's ExponentiatedGradient).
        X_data : pd.DataFrame
            The features DataFrame used for training (to get column names).
        title_prefix : str, default=""
            Prefix for the plot title (e.g., "Baseline " or "Fair ").
        """
        importances = None
        if hasattr(model, "coef_"):
            # For linear models, coef_ might be a single value if only one feature exists.
            # Ensure it's treated as an array for Series conversion.
            if isinstance(model.coef_, (int, float)):
                importances = pd.Series([model.coef_], index=X_data.columns)
            else:
                importances = pd.Series(model.coef_.flatten(), index=X_data.columns)
        elif hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_data.columns)
        elif hasattr(model, "estimators_"): # For ensemble models like those wrapped by fairlearn.reductions
            # Try to get feature importances/coefficients from the base estimator(s)
            all_importances_or_coefs = []
            for estimator in model.estimators_:
                # Check if the estimator itself has feature_importances_ or coef_
                if hasattr(estimator, "feature_importances_"):
                    all_importances_or_coefs.append(pd.Series(estimator.feature_importances_, index=X_data.columns))
                elif hasattr(estimator, "coef_"):
                    all_importances_or_coefs.append(pd.Series(estimator.coef_.flatten(), index=X_data.columns))

            if all_importances_or_coefs:
                # Average the importances/coefficients from all base estimators
                importances = pd.concat(all_importances_or_coefs, axis=1).mean(axis=1)
            else:
                print(f"\n⚠️ Base estimator importances/coefficients not available for {title_prefix.strip()} model.")

        if importances is None or importances.empty:
            print(f"\n⚠️ Feature importance not available or empty for the {title_prefix.strip()} model.")
            return

        plt.figure(figsize=(8, 6))
        importances.sort_values().plot(kind="barh", color="teal")
        plt.title(f"{title_prefix}Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()


    def run_fairness_comparison_pipeline(self, smoker_column: str = 'smoker'):
        """
        Execute the full pipeline for training a baseline model, a Fairlearn-constrained model,
        and a hybrid subgroup model. Compares their performance and fairness across sensitive
        subgroups, providing detailed feature-wise comparative tables.

        Parameters
        ----------
        smoker_column : str, default='smoker'
            The column name representing smoker status, used for the hybrid subgroup model.
        """
        # --- Initial Data Splitting for all models ---
        print("🔧 Step 0: Initial Data Splitting (Stratified if possible)...")
        self.stratified_split()
        if self.X_train is None or self.X_test is None:
            print("❌ Data splitting failed or resulted in empty sets. Cannot proceed.")
            return

        all_groupwise_metrics_mae = {}
        all_groupwise_metrics_mse = {}
        all_groupwise_metrics_r2 = {}

        # Identify features for fairness analysis across all models.
        # This is based on the full dataset (self.df) to capture all potential features.
        features_for_fairness_analysis = [
            feat for feat in self.df.columns
            if feat != self.target and (
                self.df[feat].dtype == 'object' or
                pd.api.types.is_categorical_dtype(self.df[feat]) or
                (pd.api.types.is_numeric_dtype(self.df[feat]) and self.df[feat].nunique() < 20)
            )
        ]
        # Ensure the 'smoker_column' is always included in the analysis list.
        if smoker_column not in features_for_fairness_analysis and smoker_column in self.df.columns:
            features_for_fairness_analysis.insert(0, smoker_column) # Add it to the front


        # =================================================================================
        # Step 1: Train and Evaluate Baseline Model (Standard XGBoost)
        # =================================================================================
        display(Markdown("## **Step 1: Baseline Model Performance (Standard XGBoost)** 📊"))
        print("\n📈 Training Baseline Model...")
        baseline_model = self.train_model() # Train
        
        print("📊 Evaluating Baseline Model (Overall Test Set)...")
        y_pred_train_baseline, baseline_train_scores = self.evaluate_model(baseline_model, self.X_train, self.y_train)
        y_pred_test_baseline, baseline_test_scores = self.evaluate_model(baseline_model, self.X_test, self.y_test)
        
        # Print overall test scores for Baseline Model
        print(f"Overall Test RMSE: {baseline_test_scores['rmse']:.2f}")
        print(f"Overall Test MAE: {baseline_test_scores['mae']:.2f}")
        print(f"Overall Test R2: {baseline_test_scores['r2']:.2f}")

        # Store for dynamic upper bound in fair model
        self.scores = baseline_train_scores 

        print("\n--- Visualizing Baseline Model Performance ---")
        self.plot_predictions(self.y_train, y_pred_train_baseline, self.y_test, y_pred_test_baseline, title_prefix="Baseline ")
        self.plot_feature_importance(baseline_model, self.X_train, title_prefix="Baseline ")
        
        # Collect group-wise metrics for Baseline Model
        print("\n--- Collecting Group-wise Metrics for Baseline Model ---")
        baseline_group_metrics = {}
        for feature in features_for_fairness_analysis:
            current_sensitive_groups_test = self.df.loc[self.X_test.index, feature]
            metrics_df = self.compute_groupwise_metrics(self.y_test, y_pred_test_baseline, current_sensitive_groups_test)
            baseline_group_metrics[feature] = metrics_df
        print("✅ Baseline Model Evaluation Complete.")


        # =================================================================================
        # Step 2: Train and Evaluate Fairlearn-Constrained Model
        # =================================================================================
        display(Markdown("## **Step 2: Fairlearn-Constrained Model** ⚖️"))
        fair_model = self.train_fair_model(upper_bound_val=None) # Train with dynamic bound
        
        print("\n📊 Evaluating Fairlearn-Constrained Model (Overall Test Set)...")
        y_pred_train_fair, fair_train_scores = self.evaluate_model(fair_model, self.X_train, self.y_train)
        y_pred_test_fair, fair_test_scores = self.evaluate_model(fair_model, self.X_test, self.y_test)
        
        # Print overall test scores for Fairlearn Model
        print(f"Overall Test RMSE: {fair_test_scores['rmse']:.2f}")
        print(f"Overall Test MAE: {fair_test_scores['mae']:.2f}")
        print(f"Overall Test R2: {fair_test_scores['r2']:.2f}")

        print("\n--- Visualizing Fairlearn-Constrained Model Performance ---")
        self.plot_predictions(self.y_train, y_pred_train_fair, self.y_test, y_pred_test_fair, title_prefix="Fairlearn ")
        self.plot_feature_importance(fair_model, self.X_train, title_prefix="Fairlearn ")

        # Collect group-wise metrics for Fairlearn Model
        print("\n--- Collecting Group-wise Metrics for Fairlearn-Constrained Model ---")
        fair_group_metrics = {}
        for feature in features_for_fairness_analysis:
            current_sensitive_groups_test = self.df.loc[self.X_test.index, feature]
            metrics_df = self.compute_groupwise_metrics(self.y_test, y_pred_test_fair, current_sensitive_groups_test)
            fair_group_metrics[feature] = metrics_df
        print("✅ Fairlearn Model Evaluation Complete.")


        # =================================================================================
        # Step 3: Train and Evaluate Hybrid Subgroup Model (`SubGrp`)
        # =================================================================================
        display(Markdown("## **Step 3: Hybrid Subgroup Model** 🎯"))
        print("\n📈 Training Hybrid Subgroup Model...")
        # Instantiate SubGrp with the base model and relevant columns
        subgrp_model = SubGrp(
            model=clone(self.model), # Pass a fresh clone of the base model
            smoker_column=smoker_column,
            target=self.target,
            stratify_by=self.stratify_by, # Use ModelTrainer's stratify_by for internal splits
            sensitive_features_smoker_subgroup=self.sensitive_features, # Re-use ModelTrainer's sensitive features for fair model for smokers
            test_size=self.test_size,
            random_state=self.random_state
        )
        # Fit the hybrid subgroup model on the full training data
        subgrp_model.fit(self.X_train.copy(), self.y_train.copy()) # Pass copies to avoid modifying original X_train/y_train

        print("\n📊 Evaluating Hybrid Subgroup Model (Overall Test Set)...")
        # Get predictions for both train and test for the SubGrp model
        y_pred_train_subgrp = subgrp_model.predict(self.X_train.copy()) # New line for train predictions
        y_pred_test_subgrp = subgrp_model.predict(self.X_test.copy()) # Predict on test data
        
        # Overall scores for SubGrp 
        subgrp_overall_test_scores = {
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred_test_subgrp)),
            "mae": mean_absolute_error(self.y_test, y_pred_test_subgrp),
            "r2": r2_score(self.y_test, y_pred_test_subgrp),
        }
        # Print overall test scores for SubGrp Model
        print(f"Overall Test RMSE: {subgrp_overall_test_scores['rmse']:.2f}")
        print(f"Overall Test MAE: {subgrp_overall_test_scores['mae']:.2f}")
        print(f"Overall Test R2: {subgrp_overall_test_scores['r2']:.2f}")


        print("\n--- Visualizing Hybrid Subgroup Model Performance ---")
        # Pass both train and test predictions for accurate plotting and line equations
        self.plot_predictions(self.y_train, y_pred_train_subgrp, # Corrected: use y_pred_train_subgrp
                              self.y_test, y_pred_test_subgrp, title_prefix="Subgroup ")
        # Feature importance for SubGrp is not straightforward, as it's a composite model.
        # You might need to inspect the individual `non_smoker_model` and `smoker_model` attributes
        # of the `subgrp_model` for their importances if needed.
        print("\nNote: Feature importance for the composite Hybrid Subgroup Model is not directly plotted here.")
        print("To inspect individual subgroup model importances, access `subgrp_model.non_smoker_model` and `subgrp_model.smoker_model`.")


        # Collect group-wise metrics for Hybrid Subgroup Model
        print("\n--- Collecting Group-wise Metrics for Hybrid Subgroup Model ---")
        subgrp_group_metrics = {}
        for feature in features_for_fairness_analysis:
            current_sensitive_groups_test = self.df.loc[self.X_test.index, feature]
            metrics_df = self.compute_groupwise_metrics(self.y_test, y_pred_test_subgrp, current_sensitive_groups_test)
            subgrp_group_metrics[feature] = metrics_df
        print("✅ Hybrid Subgroup Model Evaluation Complete.")


        # =================================================================================
        # Step 4: Comparative Statement (Feature-wise Metrics)
        # =================================================================================
        display(Markdown("## **Step 4: Comparative Feature-wise Analysis** 📈⚖️🎯"))
        print("Comparing MAE, MSE, and R² across Baseline, Fairlearn, and Hybrid Subgroup Models:")

        metric_names = {"MAE": "Mean Absolute Error", "MSE": "Mean Squared Error", "R2": "R-squared"}

        for metric_key, metric_full_name in metric_names.items():
            display(Markdown(f"### Comparative {metric_full_name} by Feature"))
            
            # Prepare data for this metric
            comparison_data = []
            for feature in features_for_fairness_analysis:
                # Extract the specific metric for each model and each group (including Mean/StdDev rows)
                baseline_feature_metrics = baseline_group_metrics[feature]
                fair_feature_metrics = fair_group_metrics[feature]
                subgrp_feature_metrics = subgrp_group_metrics[feature]

                for idx, row in baseline_feature_metrics.iterrows():
                    group_name = row['group']
                    
                    # Ensure corresponding row exists in other models' metrics
                    fair_value = fair_feature_metrics[fair_feature_metrics['group'] == group_name][metric_key].values[0] if group_name in fair_feature_metrics['group'].values else np.nan
                    subgrp_value = subgrp_feature_metrics[subgrp_feature_metrics['group'] == group_name][metric_key].values[0] if group_name in subgrp_feature_metrics['group'].values else np.nan

                    comparison_data.append({
                        'Feature': feature,
                        'Group': group_name,
                        f'Baseline {metric_key}': row[metric_key],
                        f'Fairlearn {metric_key}': fair_value,
                        f'Subgroup {metric_key}': subgrp_value
                    })
            
            # Create DataFrame for current metric
            comparison_df_metric = pd.DataFrame(comparison_data)
            display(comparison_df_metric.round(4))
            print("\n") # Add a newline for better separation between tables

        print("\n**End of Comparative Analysis.**")

        # Return all models and their predictions for further analysis if needed
        return {
            "baseline_model": baseline_model,
            "fair_model": fair_model,
            "subgrp_model": subgrp_model,
            "y_pred_test_baseline": y_pred_test_baseline,
            "y_pred_test_fair": y_pred_test_fair,
            "y_pred_test_subgrp": y_pred_test_subgrp,
        }


"""
===========================================================
SubGrp Class for Hybrid Subgroup Modeling
===========================================================

This class implements a hybrid subgroup modeling approach for regression,
specifically targeting fairness for a designated `smoker_column`.

It trains:
1. A standard (baseline) model for the non-smoker subgroup.
2. A fairness-aware model (using Fairlearn's ExponentiatedGradient
   with BoundedGroupLoss) for the smoker subgroup.

This class provides `.fit()` and `.predict()` methods, making it
compatible with scikit-learn's estimator interface.

Dependencies:
-------------
    pandas, numpy, sklearn (BaseEstimator, RegressorMixin, clone), ModelTrainer (this project's class)

Author:
-------
    Your AI Assistant
Date:
-----
    August 2025
===========================================================
"""
class SubGrp(BaseEstimator, RegressorMixin):
    """
    Implements a hybrid subgroup modeling approach for regression,
    specifically targeting fairness for a designated `smoker_column`.

    It trains:
    1. A standard (baseline) model for the non-smoker subgroup.
    2. A fairness-aware model (using Fairlearn's ExponentiatedGradient
       with BoundedGroupLoss) for the smoker subgroup.

    This class provides `.fit()` and `.predict()` methods, making it
    compatible with scikit-learn's estimator interface.

    Parameters
    ----------
    model : estimator
        A scikit-learn compatible regression model (e.g., XGBoostRegressor, LinearRegression).
        This will be cloned for use in both subgroup models.
    smoker_column : str, default='smoker'
        The name of the column in the DataFrame that indicates smoker status (e.g., 0 or 1).
    target : str, default='charges'
        Name of the target variable column.
    stratify_by : str or list of str, default = 'smoker'
        Column name(s) to use for stratified train-test splitting within each subgroup's
        internal ModelTrainer instance.
    sensitive_features_smoker_subgroup : str or list of str, optional
        Sensitive feature(s) to apply fairness constraints *within the smoker subgroup model*.
        If None, it defaults to the `stratify_by` feature(s) used for the ModelTrainer
        within the smoker subgroup. This is where you'd specify if there are other
        sensitive features *within* the smoker group you want to debias.
    test_size : float, default=0.2
        Proportion of dataset assigned to the test set for internal `ModelTrainer`s.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    non_smoker_model : estimator
        The fitted standard model for the non-smoker subgroup.
    smoker_model : fairlearn.reductions.ExponentiatedGradient
        The fitted fairness-aware model for the smoker subgroup.
    non_smoker_model_trainer : ModelTrainer
        Internal ModelTrainer instance used for the non-smoker subgroup.
    smoker_model_trainer : ModelTrainer
        Internal ModelTrainer instance used for the smoker subgroup (fairness-aware).
    _fitted : bool
        Flag indicating if the model has been fitted.
    """

    def __init__(self, model, smoker_column: str = 'smoker', target: str = 'charges',
                 stratify_by: str | list[str] = 'smoker',
                 sensitive_features_smoker_subgroup: str | list[str] | None = None,
                 test_size: float = 0.2, random_state: int = 42):
        self.model = model
        self.smoker_column = smoker_column
        self.target = target
        self.stratify_by = stratify_by
        self.sensitive_features_smoker_subgroup = sensitive_features_smoker_subgroup
        self.test_size = test_size
        self.random_state = random_state

        self.non_smoker_model_trainer: ModelTrainer | None = None
        self.smoker_model_trainer: ModelTrainer | None = None
        self.non_smoker_model = None # Stores the fitted base model for non-smokers
        self.smoker_model = None    # Stores the fitted fairlearn model for smokers
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the hybrid subgroup model.

        Segments the data by 'smoker_column', trains a standard model on non-smokers,
        and a fairness-aware model on smokers.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for training. Must include the `smoker_column`.
        y : pd.Series
            Target variable for training.

        Returns
        -------
        self
        """
        if self.smoker_column not in X.columns:
            raise ValueError(f"Smoker column '{self.smoker_column}' not found in the input features (X).")

        # Combine X and y for easier segmentation
        full_df = X.copy()
        full_df[self.target] = y

        # 1. Segment Data by 'Smoker' Status
        df_smoker = full_df[full_df[self.smoker_column] == 1].copy()
        df_non_smoker = full_df[full_df[self.smoker_column] == 0].copy()

        if df_smoker.empty:
            raise ValueError("No data found for the 'smoker' subgroup (smoker=1). Cannot train a fair model for them.")
        if df_non_smoker.empty:
            raise ValueError("No data found for the 'non-smoker' subgroup (smoker=0).")

        print("🚀 Starting Hybrid Subgroup Model Training:")
        print(f"Non-smoker data points: {len(df_non_smoker)}")
        print(f"Smoker data points: {len(df_smoker)}")
        print("-" * 50)

        # Before passing to ModelTrainer, ensure the smoker_column is dropped if it's not a feature
        # that the *internal base models* should learn from. This is a common practice for
        # direct subgrouping where the smoker status itself is used for partitioning, not prediction
        # within the sub-model.
        
        # Determine features to pass to internal ModelTrainer instances
        # The internal ModelTrainer will handle dropping its `target` column ('charges')
        # We need to ensure that 'smoker_column' is not passed as a *feature* to the
        # internal base models (XGBoost) if its only role is for the outer `SubGrp` partitioning.
        # This prevents the sub-models from learning 'smoker' as a feature when they already operate
        # within a smoker-defined subset.
        features_for_internal_models = [col for col in full_df.columns if col not in [self.target, self.smoker_column]]

        # Prepare dfs for ModelTrainer, ensuring 'smoker_column' is kept for initial stratification
        # if stratify_by includes it, but not necessarily as a feature the *base model* will see.
        
        # 2. Train a Standard (Baseline) Model for Non-Smokers
        print("\n--- Training Standard Model for NON-SMOKERS ---")
        # df for ModelTrainer needs all original columns + target for its internal split/feature handling
        self.non_smoker_model_trainer = ModelTrainer(
            df=df_non_smoker,
            model=clone(self.model), # Clone base model for non-smoker trainer
            target=self.target,
            stratify_by=self.stratify_by, # Use the global stratify_by for internal splits
            sensitive_features=None, # No explicit fairness for non-smokers (handled by main fairlearn model)
            test_size=self.test_size,
            random_state=self.random_state
        )
        # Run pipeline to ensure data split and evaluation for non-smoker group internally
        self.non_smoker_model_trainer.stratified_split()
        # Train baseline on its internal X_train/y_train
        self.non_smoker_model = self.non_smoker_model_trainer.train_model()
        # Evaluate to populate internal scores, needed if dynamic upper_bound were used by this trainer
        _, self.non_smoker_model_trainer.scores = self.non_smoker_model_trainer.evaluate_model(
            self.non_smoker_model,
            self.non_smoker_model_trainer.X_train,
            self.non_smoker_model_trainer.y_train
        )


        # 3. Train a Fairness-Aware Model for Smokers
        print("\n--- Training FAIRNESS-AWARE Model for SMOKERS ---")
        self.smoker_model_trainer = ModelTrainer(
            df=df_smoker,
            model=clone(self.model), # Clone base model for smoker trainer
            target=self.target,
            stratify_by=self.stratify_by, # Use the global stratify_by for internal splits
            # Apply sensitive features for fairness *within* the smoker subgroup.
            # If sensitive_features_smoker_subgroup is None, it defaults to stratify_by of this internal ModelTrainer.
            sensitive_features=self.sensitive_features_smoker_subgroup,
            test_size=self.test_size,
            random_state=self.random_state
        )
        # --- IMPORTANT FIX: Run baseline evaluation for smoker_model_trainer
        # This is needed to populate self.smoker_model_trainer.scores *before* calling train_fair_model
        self.smoker_model_trainer.stratified_split() # Ensure split first
        temp_smoker_baseline_model = self.smoker_model_trainer.train_model() # Train a temporary baseline model
        # Evaluate the temporary baseline model to populate self.smoker_model_trainer.scores
        _, self.smoker_model_trainer.scores = self.smoker_model_trainer.evaluate_model(
            temp_smoker_baseline_model,
            self.smoker_model_trainer.X_train,
            self.smoker_model_trainer.y_train
        )
        # Now train the fair model for the smoker subgroup, which can now use dynamic upper_bound
        self.smoker_model = self.smoker_model_trainer.train_fair_model(upper_bound_val=None)

        self._fitted = True
        print("\n✅ Hybrid Subgroup Model Training Complete!")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the trained subgroup models.

        Combines predictions from the non-smoker model for non-smoker instances
        and the fairness-aware model for smoker instances.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction. Must include the `smoker_column`.

        Returns
        -------
        np.ndarray
            Predicted target values, ordered corresponding to the input `X`.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if self.smoker_column not in X.columns:
            raise ValueError(f"Smoker column '{self.smoker_column}' not found in the input features (X) for prediction.")

        # Create a Series to store predictions, maintaining the original index
        y_pred = pd.Series(np.zeros(len(X)), index=X.index, dtype=float)

        # Identify smoker and non-smoker instances in the input data
        smoker_filter = X[self.smoker_column] == 1
        non_smoker_filter = X[self.smoker_column] == 0

        # Features to pass to the internal sub-models.
        # The internal ModelTrainer's X_train will *not* contain the target column.
        # We explicitly drop the smoker_column from the X passed to internal models as it's the partitioning key.
        #cols_for_sub_models = [col for col in X.columns if col != self.smoker_column]
        cols_for_sub_models = [col for col in X.columns if col != self.target]

        # Predict for non-smoker subgroup
        if non_smoker_filter.any():
            X_non_smoker_subset = X.loc[non_smoker_filter, cols_for_sub_models]
            y_pred.loc[non_smoker_filter] = self.non_smoker_model.predict(X_non_smoker_subset)
        else:
            print("No non-smoker data points in the input for prediction.")

        # Predict for smoker subgroup using the fair model
        if smoker_filter.any():
            X_smoker_subset = X.loc[smoker_filter, cols_for_sub_models]
            y_pred.loc[smoker_filter] = self.smoker_model.predict(X_smoker_subset)
        else:
            print("No smoker data points in the input for prediction.")

        return y_pred.values # Return as NumPy array for scikit-learn compatibility

    def evaluate_combined_model(self, X_eval: pd.DataFrame, y_eval: pd.Series):
        """
        Evaluates the combined hybrid model on a given dataset (e.g., test set).

        Computes overall metrics (RMSE, MAE, R²) and then provides a feature-wise
        diagnostic breakdown using the `ModelTrainer`'s utility.

        Parameters
        ----------
        X_eval : pd.DataFrame
            Feature matrix for evaluation. Must include the `smoker_column`.
        y_eval : pd.Series
            True target values for evaluation.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            A tuple containing:
            - overall_metrics_df: DataFrame with RMSE, MAE, R2.
            - all_groupwise_metrics: Dictionary with metrics per feature/group.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if self.smoker_column not in X_eval.columns:
            raise ValueError(f"Smoker column '{self.smoker_column}' not found in the evaluation features (X_eval).")

        print("\n--- Evaluating Combined Hybrid Model ---")
        y_pred_combined = self.predict(X_eval)

        # Overall Metrics
        overall_rmse = np.sqrt(mean_squared_error(y_eval, y_pred_combined))
        overall_mae = mean_absolute_error(y_eval, y_pred_combined)
        overall_r2 = r2_score(y_eval, y_pred_combined)

        overall_metrics_df = pd.DataFrame({
            "Metric": ["RMSE", "MAE", "R²"],
            "Value": [overall_rmse, overall_mae, overall_r2]
        }).set_index("Metric")
        display(Markdown("### Overall Combined Model Performance"))
        display(overall_metrics_df.round(4))

        # Group-wise Metrics (re-using ModelTrainer's utility)
        # Create a dummy ModelTrainer to use its compute_groupwise_metrics method
        # We pass a minimal df since it only needs y_true, y_pred, sensitive_groups
        dummy_trainer = ModelTrainer(df=pd.DataFrame(), model=self.model, target=self.target)

        # Automatically select all categorical features and low-cardinality numerical features
        # for fairness analysis, excluding the target column.
        features_for_fairness_analysis = [
            feat for feat in X_eval.columns
            if (feat != self.target) and (
                X_eval[feat].dtype == 'object' or
                pd.api.types.is_categorical_dtype(X_eval[feat]) or
                (pd.api.types.is_numeric_dtype(X_eval[feat]) and X_eval[feat].nunique() < 20)
            )
        ]
        # Ensure the smoker_column is included in the analysis if it's not already
        if self.smoker_column not in features_for_fairness_analysis:
            features_for_fairness_analysis.insert(0, self.smoker_column) # Prioritize smoker column

        display(Markdown("### Feature-wise Diagnostic Breakdown (Fairness Analysis for Combined Model) 🕵️‍♀️"))
        all_groupwise_metrics = {}
        for feature in features_for_fairness_analysis:
            display(Markdown(f"#### Analyzing by Feature: **`{feature}`**"))
            current_sensitive_groups = X_eval[feature]
            group_metrics = dummy_trainer.compute_groupwise_metrics(y_eval, y_pred_combined, current_sensitive_groups)
            display(group_metrics.round(4))
            all_groupwise_metrics[feature] = group_metrics
            display(Markdown("---"))

        print("\nEvaluation of Combined Hybrid Model Complete! ✅")
        return overall_metrics_df, all_groupwise_metrics
