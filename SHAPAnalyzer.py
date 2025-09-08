"""
===========================================================
SHAPAnalyzer - SHAP Value Analysis Utility
===========================================================

This module implements a single class, `SHAPAnalyzer`, which encapsulates
end-to-end SHAP analysis for both standard scikit-learn style models and
more complex wrapped models such as Fairlearn's `ExponentiatedGradient`.

Key Features:
-------------
1. **Automatic Model Type Handling**
   - Supports tree models, general ML models, and Fairlearn's wrapped estimators.
   - Chooses the appropriate SHAP Explainer (TreeExplainer or KernelExplainer).

2. **Global Feature Importance**
   - Computes and displays mean absolute SHAP values.
   - Generates SHAP beeswarm plots for visual interpretation.

3. **Interaction Analysis** (Tree models only)
   - Computes SHAP interaction values.
   - Displays ranked interaction tables, half-matrix heatmaps, and targeted interaction plots.

4. **Workflow Integration**
   - Provides a `run_analysis()` method for a complete SHAP workflow.
   - Supports top-N feature display and interaction inspection.

Dependencies:
-------------
    pandas, numpy, shap, matplotlib, seaborn, IPython.display

Author:
-------
    [Your Name / Team]
Date:
-----
    August 2025
===========================================================
"""

# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

# Modeling & Evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, List, Optional, Union, Tuple
from IPython.display import display, HTML

# Initialize SHAP JS visualization support
shap.initjs()


class SHAPAnalyzer:
    """
    A SHAP analysis helper for regression and classification models.

    This class:
    - Handles standard models and Fairlearn's ExponentiatedGradient wrapper.
    - Automates explainer selection (TreeExplainer, KernelExplainer, generic Explainer).
    - Computes SHAP global feature importance and (for tree models) interaction values.
    - Creates visualizations summarizing SHAP results.

    Parameters
    ----------
    model : object
        Fitted model or estimator with a .predict() method.
    target_col : str
        Name of the target column in the input DataFrame.
    stratify_cols : str or List[str]
        Columns to stratify on when splitting data.
    test_size : float, default=0.2
        Fraction of the data to allocate to the test set.
    background_size : int, default=512
        Number of samples to use as SHAP background data.
    top_features : int, default=15
        Number of top features to show in global importance.
    group_top_n : int, default=5
        Top-N features per group category (synthetic, engineered, original).
    seed : int, default=42
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        model,
        target_col: str,
        stratify_cols: Union[str, List[str]],
        test_size: float = 0.2,
        background_size: int = 512,
        top_features: int = 15,
        group_top_n: int = 5,
        seed: int = 42,
    ):
        # Store parameters
        self.model = model
        self.target_col = target_col
        self.stratify_cols = stratify_cols
        self.test_size = test_size
        self.background_size = background_size
        self.top_features = top_features
        self.group_top_n = group_top_n
        self.seed = seed

        # Internal: SHAP explainer instance
        self._explainer = None

    # =========================================================
    # Public API
    # =========================================================
    def run_analysis(
        self,
        data: pd.DataFrame,
        feature_pairs: Optional[Dict[str, str]] = None,
        top_interactions_n: int = 10
    ) -> Dict[str, object]:
        """
        Run end-to-end SHAP analysis on a dataset.

        Steps:
        ------
        1. Prepare and sample background/test data.
        2. Initialize appropriate SHAP explainer.
        3. Compute SHAP values for the test set.
        4. Generate global importance dataframe and beeswarm plot.
        5. (If supported) Compute interactions: table, heatmap, targeted plots.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing features and the target column.
        feature_pairs : dict, optional
            Mapping of feature 1 -> feature 2 for targeted interaction dependence plots.
        top_interactions_n : int, default=10
            Max number of top interactions to display.

        Returns
        -------
        dict
            Dictionary containing computed results: importance dataframe, plots, tables.
        """
        X_test, _, _ = self._prepare_data(data)
        self._warn_on_object_columns(X_test)

        # Background data for SHAP expectations
        X_background = self._sample_background(X_test)

        # Create the appropriate SHAP explainer
        self._init_explainer(X_background)

        print("Calculating SHAP values... (This may take a moment for non-tree models)")
        shap_values_obj = self._explainer(X_test)

        # Global feature importance
        global_importance_df = self._global_feature_importance(shap_values_obj)
        results = {
            "global_feature_importance": global_importance_df,
            "feature_importance_plot": self._plot_global_importance(shap_values_obj, X_test),
        }

        # Tree SHAP supports interactions
        if hasattr(self._explainer, "shap_interaction_values"):
            print("Tree explainer detected. Computing SHAP interaction values...")
            shap_interaction_values = self._compute_interactions(X_test)
            interaction_table = self._summarize_interactions(shap_interaction_values, X_test.columns)
            results["interaction_table"] = interaction_table
            results["interaction_heatmap"] = self._plot_interaction_heatmap(
                interaction_table, list(X_test.columns)
            )
            if feature_pairs:
                results["interaction_plot"] = self._plot_interactions(
                    X_test, feature_pairs, shap_interaction_values
                )
        else:
            print("Note: SHAP interaction values are not available for this model type.")

        # Display all results
        self._display_results(results, top_interactions_n)
        return results

    # =========================================================
    # Internal helpers
    # =========================================================
    def _display_results(self, results: dict, top_interactions_n: int):
        """Display all generated tables & plots inside Jupyter/IPython."""
        display(HTML("<h2>SHAP Analysis Results</h2>"))
        display(HTML("<h3>Auditing global SHAP feature importance...</h3>"))
        display(results["global_feature_importance"].head(self.top_features))
        display(HTML("<h3>Global Feature Importance Plot (Beeswarm)</h3>"))
        display(results["feature_importance_plot"])
        #if "interaction_table" in results:
            #display(HTML(f"<h3>Top {top_interactions_n} Feature Interactions (Table)</h3>"))
            #display(results["interaction_table"].head(top_interactions_n))
        #if "interaction_heatmap" in results:
            #display(HTML("<h3>Feature Interaction Heatmap</h3>"))
            #display(results["interaction_heatmap"])
        #if "interaction_plot" in results:
            #display(HTML("<h3>Requested SHAP Interaction Plots</h3>"))
            #display(results["interaction_plot"])

    def _init_explainer(self, background: pd.DataFrame):
        """
        Initialize SHAP explainer depending on the model type.

        Cases handled:
        1. Fairlearn ExponentiatedGradient wrapper -> KernelExplainer
        2. Scikit-learn tree ensembles -> TreeExplainer(interventional)
        3. Other supported models -> shap.Explainer()
        """
        if type(self.model).__name__ == 'ExponentiatedGradient':
            print("Fairlearn model detected. Using KernelExplainer with model.predict.")
            self._explainer = shap.Explainer(self.model.predict, background)
        elif hasattr(self.model, 'estimators_'):
            print("Tree ensemble detected. Using interventional TreeExplainer.")
            self._explainer = shap.TreeExplainer(
                self.model, background, feature_perturbation="interventional"
            )
        else:
            print("Standard model detected. Using default shap.Explainer().")
            self._explainer = shap.Explainer(self.model, background)

    def _get_feature_type(self, feature_name: str) -> str:
        """Classify feature origin by its naming pattern."""
        if feature_name.endswith("_syn"):
            return "Synthetic"
        if feature_name.endswith("_bin") or "__" in feature_name:
            return "Engineered"
        return "Original"

    def _global_feature_importance(self, shap_values: shap.Explanation) -> pd.DataFrame:
        """Compute and label mean absolute SHAP value per feature."""
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        imp_df = pd.DataFrame({
            "Feature": shap_values.feature_names,
            "Mean |SHAP|": mean_abs_shap
        })
        imp_df["Type"] = imp_df["Feature"].apply(self._get_feature_type)
        return imp_df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

    def _plot_interaction_heatmap(self, interaction_df: pd.DataFrame, feature_names: List[str]) -> Optional[plt.Figure]:
        """Draw upper-triangular SHAP interaction heatmap."""
        if interaction_df.empty:
            return None
        interaction_matrix = interaction_df.pivot(
            index='Parameter A', columns='Parameter B', values='interaction_value'
        )
        interaction_matrix = interaction_matrix.reindex(index=feature_names, columns=feature_names)
        interaction_matrix = interaction_matrix.add(interaction_matrix.T, fill_value=0)
        np.fill_diagonal(interaction_matrix.values, np.nan)
        vmin, vmax = 0, interaction_df['interaction_value'].max()
        mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))
        fig, ax = plt.subplots(figsize=(max(10, len(feature_names)),
                                        max(8, len(feature_names) * 0.75)))
        sns.heatmap(interaction_matrix, mask=mask, cmap="viridis",
                    annot=True, fmt=".2f", linewidths=.5, ax=ax,
                    vmin=vmin, vmax=vmax)
        ax.set_title("SHAP Interaction Value Heatmap", fontsize=16)
        plt.tight_layout()
        fig_to_return = plt.gcf()
        plt.close(fig_to_return)
        return fig_to_return

    def _summarize_interactions(self, shap_interaction_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Create ranked list of mean absolute SHAP interaction values (pairs only)."""
        mean_abs_interactions = np.abs(shap_interaction_values).mean(0)
        interaction_summary = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                interaction_summary.append({
                    "Parameter A": feature_names[i],
                    "Parameter B": feature_names[j],
                    "interaction_value": mean_abs_interactions[i, j]
                })
        return pd.DataFrame(interaction_summary).sort_values(
            "interaction_value", ascending=False
        ).reset_index(drop=True)

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
        """Prepare features/target for analysis (simple split placeholder)."""
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        return X, y, data.index

    def _sample_background(self, X: pd.DataFrame) -> pd.DataFrame:
        """Randomly sample background data for SHAP expectations."""
        return shap.sample(X, min(self.background_size, len(X)), random_state=self.seed)

    def _warn_on_object_columns(self, X: pd.DataFrame):
        """Warn if object-typed columns exist (may need encoding)."""
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            warnings.warn(f"Object columns detected (SHAP may not handle these well): {object_cols}")

    def _plot_global_importance(self, shap_values: shap.Explanation, X: pd.DataFrame) -> plt.Figure:
        """Generate SHAP beeswarm plot for global feature importance."""
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def _compute_interactions(self, X: pd.DataFrame) -> np.ndarray:
        """Request SHAP interaction values from the explainer."""
        return self._explainer.shap_interaction_values(X)

    def _plot_interactions(
        self,
        X: pd.DataFrame,
        feature_pairs: Dict[str, str],
        shap_interaction_values: np.ndarray,
        title: str = "SHAP Interaction Plots"
    ) -> Optional[plt.Figure]:
        """Generate dependence plots for specified feature pairs."""
        n = len(feature_pairs)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()
        for i, (feat1, feat2) in enumerate(feature_pairs.items()):
            ax = axes[i]
            try:
                shap.dependence_plot((feat1, feat2), shap_interaction_values, X, ax=ax, show=False)
            except Exception as e:
                ax.text(0.5, 0.5,
                        f"Error plotting {feat1} vs {feat2}:\n{e}",
                        ha='center', va='center')
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

