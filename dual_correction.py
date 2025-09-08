"""
===============================================================================
FairnessByDualCorrection Module
===============================================================================
Purpose:
    This module defines the FairnessByDualCorrection class for applying fairness-aware bias 
    correction to model predictions through global and groupwise additive shifts. The corrections
    minimize specified error metrics (MAE or MSE) on training data segmented by sensitive groups.
    Improvements include robust outlier removal, stable group offsets computation, and tuning of 
    correction strength. Audit capabilities are included to track fairness impact by group.
===============================================================================
"""

import pandas as pd
import numpy as np


class FairnessByDualCorrection:
    """
    Applies fairness corrections to predictions by learning additive shifts. 
    
    Supports global bias correction and group-level adjustments based on sensitive features.
    Corrections minimize MAE or MSE objective between predictions and true values.
    
    Attributes:
    -----------
    apply_global : bool
        Whether to apply a single global shift to all predictions.
    apply_groupwise : bool
        Whether to apply additional group-specific shifts.
    objective : str
        Error metric to minimize, either 'mae' or 'mse'.
    combine_sensitive : bool
        If True, sensitive features are combined into a single group key.
    lambda_strength : float
        Strength of correction applied (scaling factor for learned offsets).
    min_group_size : int
        Minimum number of samples per group to compute stable group offsets.
    z_thresh : float
        Z-score threshold for removing outliers.
    remove_outliers : bool
        Whether to remove outliers before fitting.
    global_shift : float
        Learned global additive offset.
    group_offsets : dict
        Learned group-specific additive offsets relative to global_shift.
    sensitive_cols : list
        Names of sensitive feature columns to consider.
    audit_log : dict
        Stores before/after audit summaries by sensitive group.
    """

    def __init__(
        self,
        apply_global=True,
        apply_groupwise=True,
        objective='mae',
        combine_sensitive=True,
        lambda_strength=0.5,  # Start with half correction strength; tune empirically
        min_group_size=3,    # Increased min group size for stability
        z_thresh=2.5,         # More aggressive outlier removal threshold
        remove_outliers=True  # Enable outlier removal by default
    ):
        """
        Initialize fairness correction parameters and state.
        
        Parameters
        ----------
        apply_global : bool, default=True
            Whether to apply a global correction.
        apply_groupwise : bool, default=True
            Whether to apply group-wise corrections.
        objective : str, default='mae'
            Metric to minimize for offsets. Options: 'mae', 'mse'.
        combine_sensitive : bool, default=True
            Combine sensitive feature values to single group tag if True.
        lambda_strength : float, default=0.5
            Multiplier scaling the learned offsets applied to predictions.
        min_group_size : int, default=10
            Minimum group size to compute offsets reliably.
        z_thresh : float, default=2.5
            Z-score threshold for outlier removal.
        remove_outliers : bool, default=True
            If True, perform outlier removal on targets before fitting.
        """
        print("[FairnessByDualCorrection] Initialized with lambda_strength=%.3f, min_group_size=%d, z_thresh=%.2f, remove_outliers=%s" %
              (lambda_strength, min_group_size, z_thresh, remove_outliers))

        self.apply_global = apply_global
        self.apply_groupwise = apply_groupwise
        self.objective = objective.lower()
        self.combine_sensitive = combine_sensitive
        self.lambda_strength = lambda_strength
        self.min_group_size = min_group_size
        self.z_thresh = z_thresh
        self.remove_outliers = remove_outliers

        # Initialize variables for learned offsets
        self.global_shift = 0.0
        self.group_offsets = {}
        self.sensitive_cols = []
        self.audit_log = {}

    def _remove_outliers(self, df, col):
        """
        Remove outliers from a DataFrame column based on z-score threshold.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data containing target values.
        col : str
            Column name for which to calculate z-scores.
            
        Returns
        -------
        pd.DataFrame
            DataFrame filtered to exclude outliers beyond z_thresh.
        """
        if not self.remove_outliers:
            #print("[_remove_outliers] Outlier removal disabled, proceeding without filtering.")
            return df
        
        z_scores = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
        filtered_df = df[abs(z_scores) <= self.z_thresh]
        #print(f"[_remove_outliers] Removed {len(df) - len(filtered_df)} outliers using z_thresh={self.z_thresh}.")
        return filtered_df

    def _calc_offset(self, y_true, y_pred):
        """
        Calculate additive offset minimizing chosen error metric.
        
        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Model predicted values.
            
        Returns
        -------
        float
            Calculated offset to add to predictions.
        """
        if self.objective == 'mae':
            offset = float(np.median(y_true - y_pred))
        elif self.objective == 'mse':
            offset = float(np.mean(y_true - y_pred))
        else:
            raise ValueError("objective must be 'mae' or 'mse'")
        
        #print(f"[_calc_offset] Calculated {self.objective.upper()} offset: {offset:.4f}")
        return offset
        
    def _make_tags(self, df):
        """
        Create group tags by combining sensitive feature values or separately.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data with sensitive feature columns.
            
        Returns
        -------
        pd.Series or dict
            Combined tags if combine_sensitive=True else dict of series per feature.
        """
        if self.combine_sensitive:
            tags = df[self.sensitive_cols].astype(str).agg('_'.join, axis=1)
            #print(f"[_make_tags] Created combined group tags for {len(tags.unique())} groups.")
            return tags
        else:
            tags_dict = {col: df[col].astype(str) for col in self.sensitive_cols}
            #print(f"[_make_tags] Created separate group tags for sensitive features: {self.sensitive_cols}")
            return tags_dict
        
    def fit_from_dataframe(self, df, target_col, pred_col, sensitive_cols,
                           split_col='split', train_tag='train'):
        """
        Fit correction shifts from training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data containing target, predictions, sensitive features, and split info.
        target_col : str
            Column with true target values.
        pred_col : str
            Column with model predictions.
        sensitive_cols : list
            List of sensitive columns to base group offsets on.
        split_col : str, default='split'
            Column indicating train/test splits.
        train_tag : str, default='train'
            Value in split_col denoting training rows.
            
        Returns
        -------
        tuple
            (train_tags, test_tags) group identifiers for training/testing sets.
        """
        self.sensitive_cols = sensitive_cols
        self.audit_log.clear()
        
        #print("[fit_from_dataframe] Splitting training and test datasets.")
        train_df = df[df[split_col] == train_tag].copy()
        test_df = df[df[split_col] != train_tag].copy()
        
        train_df = self._remove_outliers(train_df, target_col)
        
        y_true_train = train_df[target_col]
        y_pred_train = train_df[pred_col]
        
        # Calculate global offset
        #print("[fit_from_dataframe] Calculating global offset...")
        self.global_shift = self._calc_offset(y_true_train, y_pred_train)
        #print(f"[fit_from_dataframe] Global shift set to {self.global_shift:.4f}")
        
        # Compute residuals after applying global shift scaled by lambda_strength
        adjusted_preds = y_pred_train + self.lambda_strength * self.global_shift
        
        if self.apply_groupwise:
            #print("[fit_from_dataframe] Calculating group-wise offsets...")
            self.group_offsets.clear()
            
            if self.combine_sensitive:
                tags_train = self._make_tags(train_df)
                train_df["__tag__"] = tags_train
                
                self.group_offsets["__tag__"] = {}
                
                for group_val, gdata in train_df.groupby("__tag__"):
                    if len(gdata) < self.min_group_size:
                        #print(f"  [fit_from_dataframe] Skipping group '{group_val}' due to size {len(gdata)} < {self.min_group_size}")
                        continue
                    
                    group_offset = self._calc_offset(gdata[target_col],
                                                     gdata[pred_col] + self.lambda_strength * self.global_shift)
                    
                    relative_offset = group_offset - self.global_shift
                    self.group_offsets["__tag__"][group_val] = relative_offset
                    #print(f"  [fit_from_dataframe] Group '{group_val}' offset: {relative_offset:.4f}")
            else:
                for col in self.sensitive_cols:
                    self.group_offsets[col] = {}
                    for group_val, gdata in train_df.groupby(col):
                        if len(gdata) < self.min_group_size:
                            #print(f"  [fit_from_dataframe] Skipping group '{group_val}' in '{col}' due to size {len(gdata)} < {self.min_group_size}")
                            continue
                        
                        group_offset = self._calc_offset(gdata[target_col],
                                                         gdata[pred_col] + self.lambda_strength * self.global_shift)
                        
                        relative_offset = group_offset - self.global_shift
                        self.group_offsets[col][group_val] = relative_offset
                        #print(f"  [fit_from_dataframe] Group '{group_val}' in '{col}' offset: {relative_offset:.4f}")
            #print("[fit_from_dataframe] Groupwise offsets calculation complete.")
        
        if self.combine_sensitive:
            return self._make_tags(train_df), self._make_tags(test_df)
        else:
            return (
                {col: train_df[col] for col in self.sensitive_cols},
                {col: test_df[col] for col in self.sensitive_cols}
            )

    def transform(self, y_pred_test, group_test):
        """
        Apply learned fairness corrections to predicted values.
        
        Parameters
        ----------
        y_pred_test : pd.Series
            Initial model predictions to correct.
        group_test : pd.Series or dict
            Group tags for each prediction.
            
        Returns
        -------
        pd.Series
            Corrected predictions after applying global and group offsets.
        """
        corrected = y_pred_test.copy()
        
        if self.apply_global:
            #print(f"[transform] Applying global shift scaled by lambda_strength {self.lambda_strength}")
            corrected += self.lambda_strength * self.global_shift
        
        if self.apply_groupwise:
            if self.combine_sensitive:
                tags = pd.Series(group_test, index=corrected.index)
                offsets = tags.map(self.group_offsets.get("__tag__", {}))
                #print("[transform] Applying combined group-wise offsets")
                corrected += self.lambda_strength * offsets.fillna(0.0)
            else:
                #print("[transform] Applying per-feature group-wise offsets")
                for col in self.sensitive_cols:
                    tags = pd.Series(group_test[col], index=corrected.index)
                    offsets = tags.map(self.group_offsets.get(col, {}))
                    corrected += self.lambda_strength * offsets.fillna(0.0)
        
        return corrected
    
    def transform_with_audit(self, y_pred_test_before, group_test):
        """
        Apply corrections and generate fairness audit logs showing before/after means.
        
        Parameters
        ----------
        y_pred_test_before : pd.Series
            Original predictions before correction.
        group_test : pd.Series or dict
            Group tags for each prediction.
            
        Returns
        -------
        tuple
            (corrected_predictions: pd.Series, audit_log: dict)
        """
        corrected_preds = self.transform(y_pred_test_before.copy(), group_test)
        self.audit_log.clear()
        
        if self.combine_sensitive:
            tags_split = pd.Series(group_test).str.split('_', expand=True)
            for i, col in enumerate(self.sensitive_cols):
                values = sorted(tags_split[i].unique())
                audit_lines = [f"Fairness Audit for {col}:"]
                for val in values:
                    submask = tags_split[i] == val
                    mean_before = y_pred_test_before[submask.values].mean()
                    mean_after = corrected_preds[submask.values].mean()
                    audit_lines.append(f"- {col}={val} : before = {mean_before:.2f}, after = {mean_after:.2f}")
                self.audit_log[col] = "\n".join(audit_lines)
        else:
            for col in self.sensitive_cols:
                unique_groups = sorted(pd.Series(group_test[col]).unique())
                audit_lines = [f"Fairness Audit for {col}:"]
                for grp in unique_groups:
                    mask = pd.Series(group_test[col]) == grp
                    mean_before = y_pred_test_before[mask.values].mean()
                    mean_after = corrected_preds[mask.values].mean()
                    audit_lines.append(f"- {col}={grp} : before = {mean_before:.2f}, after = {mean_after:.2f}")
                self.audit_log[col] = "\n".join(audit_lines)
        
        #print("[transform_with_audit] Fairness audit report generated.")
        return corrected_preds, self.audit_log










"""
===============================================================================
FairnessByAddingError Module
===============================================================================
Purpose:
    Implements a fairness-aware correction by learning per-sample additive shifts
    to equalize subgroup Mean Absolute Error (MAE) in model predictions. Corrections
    selectively pull predictions closer to true values in underperforming subgroups,
    while avoiding worsening performance in better-performing subgroups.
    
    Provides an interface compatible with FairnessByDualCorrection for seamless usage.
===============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class FairnessByTrippleCorrection:
    """
    Fairness correction by learning a linear regression of residuals using:
        1. (Subgroup MAE - Global MAE) for each sensitive group,
        2. A global slope and intercept,
        3. Subgroup-specific slopes and intercepts.

    Correction is then applied to test predictions using this learned model.
    
    Attributes:
    -----------
    apply_global : bool
        Whether to apply a global trend correction.
    apply_groupwise : bool
        Whether to learn and apply group-specific slopes/biases.
    combine_sensitive : bool
        Whether to use combined sensitive groups or per-feature.
    lambda_strength : float
        Controls magnitude of applied correction during transform.
    min_group_size : int
        Minimum rows per group for robust regression fit.
    sensitive_cols : list
        Names of sensitive feature columns used for grouping.
    reg_dict : dict
        Fitted LinearRegression objects for each group.
    global_reg : LinearRegression
        Fitted regression for global corrections.
    group_mae_diff : dict
        Stores (group, MAE - global MAE) mapping.
    audit_log : dict
        Stores fairness audit strings.
    """
    def __init__(self,
                 apply_global=True,
                 apply_groupwise=True,
                 combine_sensitive=True,
                 lambda_strength=1.0,
                 min_group_size=3):
        print(f"[TrippleFactorCorrection] Initialized. lambda_strength={lambda_strength}, min_group_size={min_group_size}")
        self.apply_global = apply_global
        self.apply_groupwise = apply_groupwise
        self.combine_sensitive = combine_sensitive
        self.lambda_strength = lambda_strength
        self.min_group_size = min_group_size
        self.sensitive_cols = []
        self.reg_dict = {}
        self.global_reg = None
        self.group_mae_diff = {}
        self.audit_log = {}

    def _make_tags(self, df):
        if self.combine_sensitive:
            tags = df[self.sensitive_cols].astype(str).agg('_'.join, axis=1)
            #print(f"[_make_tags] Combined group tags: {len(tags.unique())} groups.")
            return tags
        else:
            tags_dict = {col: df[col].astype(str) for col in self.sensitive_cols}
            #print(f"[_make_tags] Separate group tags for features: {self.sensitive_cols}.")
            return tags_dict

    def fit_from_dataframe(self, df, target_col, pred_col, sensitive_cols,
                           split_col='split', train_tag='train'):
        self.sensitive_cols = sensitive_cols
        self.audit_log.clear()
        train_df = df[df[split_col] == train_tag].copy()
        test_df = df[df[split_col] != train_tag].copy()
        train_df['error'] = train_df[target_col] - train_df[pred_col]  # residual
        train_df['abs_error'] = train_df['error'].abs()

        print("[TrippleFactorCorrection] Fitting regression-based correction...")

        # Build group tags
        if self.combine_sensitive:
            train_df['__tag__'] = self._make_tags(train_df)
            group_tag_col = '__tag__'
        else:
            group_tag_col = self.sensitive_cols[0]

        # Compute global MAE
        global_mae = train_df['abs_error'].mean()
        print(f"[TrippleFactorCorrection] Overall train MAE: {global_mae:.4f}")
        self.global_mae = global_mae

        # Compute each group's MAE + MAE_diff
        group_maes = train_df.groupby(group_tag_col)['abs_error'].mean().to_dict()
        self.group_mae_diff = {g: (m - global_mae) for g, m in group_maes.items()}

        # --- Fit global regression (may or may not be used depending on apply_global) ---
        train_df['_global_mae_diff'] = train_df[group_tag_col].map(self.group_mae_diff)
        global_X = train_df[['_global_mae_diff']]
        global_y = train_df['error']
        self.global_reg = LinearRegression().fit(global_X, global_y)
        print(f"[TrippleFactorCorrection] Fitted global regression: coef={self.global_reg.coef_[0]:.4f}, intercept={self.global_reg.intercept_:.4f}")

        # --- Fit per-group regression ---
        self.reg_dict = {}
        for g, group_df in train_df.groupby(group_tag_col):
            if len(group_df) < self.min_group_size:
                #print(f"  [TrippleFactorCorrection] Group '{g}': skipped (size {len(group_df)})")
                continue
            # For group g: fit error ~ (group_MAE - global_MAE)
            Xg = group_df[['_global_mae_diff']]
            yg = group_df['error']
            reg = LinearRegression().fit(Xg, yg)
            self.reg_dict[g] = reg
            #print(f"  [TrippleFactorCorrection] Group '{g}' regression: coef={reg.coef_[0]:.4f}, intercept={reg.intercept_:.4f} (n={len(Xg)})")

        # Map tags for test/train return as usual
        if self.combine_sensitive:
            return train_df['__tag__'], self._make_tags(test_df)
        else:
            return {col: train_df[col] for col in sensitive_cols}, {col: test_df[col] for col in sensitive_cols}

    def transform(self, y_pred_test, group_test):
        corrected = y_pred_test.copy()
        print(f"[TrippleFactorCorrection] Applying tripple-factor corrections to test predictions...")

        # Set up per-sample corrections
        test_index = corrected.index
        if isinstance(group_test, pd.Series):
            group_tags = group_test
        else:
            # For single-feature (non-combined), just take the first
            group_key = list(group_test.keys())[0]
            group_tags = pd.Series(group_test[group_key], index=corrected.index)

        # Map group corrections
        corrections = []
        for idx, group in group_tags.items():
            # Identify the "mae_diff" feature for regression input
            mae_diff = self.group_mae_diff.get(group, 0.0)
            #X_input = np.array([[mae_diff]])
            X_input = pd.DataFrame({'_global_mae_diff': [mae_diff]})
            # Use group regression if exists, otherwise default to global
            if self.apply_groupwise and group in self.reg_dict:
                reg = self.reg_dict[group]
                corr = reg.predict(X_input)[0]
            elif self.apply_global and self.global_reg is not None:
                corr = self.global_reg.predict(X_input)[0]
            else:
                corr = 0.0
            corrections.append(self.lambda_strength * corr)
        # Apply correction
        corrected = corrected + corrections
        return corrected

    def transform_with_audit(self, y_pred_test_before, group_test):
        corrected_preds = self.transform(y_pred_test_before.copy(), group_test)
        self.audit_log.clear()
        if self.combine_sensitive:
            tags_split = pd.Series(group_test).str.split('_', expand=True)
            for i, col in enumerate(self.sensitive_cols):
                values = sorted(tags_split[i].unique())
                audit_lines = [f"Fairness Audit for {col}:"]
                for val in values:
                    mask = tags_split[i] == val
                    mean_before = y_pred_test_before[mask.values].mean()
                    mean_after = corrected_preds[mask.values].mean()
                    audit_lines.append(f"- {col}={val}: before = {mean_before:.2f}, after = {mean_after:.2f}")
                self.audit_log[col] = "\n".join(audit_lines)
        else:
            for col in self.sensitive_cols:
                unique_groups = sorted(pd.Series(group_test[col]).unique())
                audit_lines = [f"Fairness Audit for {col}:"]
                for group_val in unique_groups:
                    mask = pd.Series(group_test[col]) == group_val
                    mean_before = y_pred_test_before[mask.values].mean()
                    mean_after = corrected_preds[mask.values].mean()
                    audit_lines.append(f"- {col}={group_val}: before = {mean_before:.2f}, after = {mean_after:.2f}")
                self.audit_log[col] = "\n".join(audit_lines)
        print("[transform_with_audit] Fairness audit report generated.")
        return corrected_preds, self.audit_log








