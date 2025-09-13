# Standard libs
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Your simulator (keep this import path consistent with your project)
from RegressionErrorSimulator import RegressionErrorSimulator


class FairnessByDualCorrection:
    """
    Dual-correction for fairness:
    - Global shift (applied to everyone)
    - Group-wise relative offsets (applied on top, per sensitive group)
    - Optional Type I/II error testing on the correction regression
    """

    def __init__(
        self,
        apply_global=True,
        apply_groupwise=True,
        objective='mae',
        combine_sensitive=True,
        lambda_strength=0.5,
        min_group_size=3,
        z_thresh=2.5,
        remove_outliers=True,
        alpha=0.05
    ):
        self.apply_global = apply_global
        self.apply_groupwise = apply_groupwise
        self.objective = objective.lower()
        self.combine_sensitive = combine_sensitive
        self.lambda_strength = float(lambda_strength)
        self.min_group_size = int(min_group_size)
        self.z_thresh = float(z_thresh)
        self.remove_outliers = bool(remove_outliers)
        self.alpha = float(alpha)

        # Learned artifacts
        self.global_shift = 0.0
        self.group_offsets = {}  # {'__tag__': {tag: rel_offset}} or {'col': {val: rel_offset}}
        self.sensitive_cols = []
        self.audit_log = {}
        self.error_summary = None
        self.error_details = None

    # ----------------------------
    # Internal utilities
    # ----------------------------
    def _remove_outliers(self, df, col):
        if not self.remove_outliers:
            return df
        s = df[col]
        z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
        return df[abs(z) <= self.z_thresh].copy()

    def _calc_offset(self, y_true, y_pred):
        # Bias estimate: robust median for MAE, mean for MSE
        diff = (y_true - y_pred)
        if self.objective == 'mae':
            return float(np.median(diff))
        elif self.objective == 'mse':
            return float(np.mean(diff))
        else:
            raise ValueError("objective must be 'mae' or 'mse'")

    def _generate_tags(self, df):
        if self.combine_sensitive:
            return df[self.sensitive_cols].astype(str).agg('_'.join, axis=1)
        else:
            return {col: df[col].astype(str) for col in self.sensitive_cols}

    def _compute_group_offsets(self, df, target_col, pred_col):
        """
        Compute relative group offsets: group_offset - global_shift,
        using predictions already adjusted by the global component.
        """
        self.group_offsets.clear()
        adjusted_pred = df[pred_col] + self.lambda_strength * self.global_shift

        if self.combine_sensitive:
            tags = self._generate_tags(df)  # pd.Series of combined tags
            df = df.copy()
            df['__tag__'] = tags
            self.group_offsets['__tag__'] = {}

            for tag, g in df.groupby('__tag__'):
                if len(g) < self.min_group_size:
                    continue
                offset = self._calc_offset(g[target_col], adjusted_pred.loc[g.index])
                self.group_offsets['__tag__'][tag] = float(offset - self.global_shift)
        else:
            for col in self.sensitive_cols:
                self.group_offsets[col] = {}
                for val, g in df.groupby(col):
                    if len(g) < self.min_group_size:
                        continue
                    offset = self._calc_offset(g[target_col], adjusted_pred.loc[g.index])
                    self.group_offsets[col][val] = float(offset - self.global_shift)

    def _run_error_testing(self, df, target_col):
        """
        Type I/II error testing on a linear model fit to the cleaned design matrix.
        Uses only the provided df (training split when called from fit_from_dataframe).
        """
        if df.empty:
            self.error_summary = {"note": "No data available for error testing."}
            self.error_details = None
            return

        # Build X/y for regression (exclude target only; keep preds and sensitive fields)
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # One-hot encode categoricals except numeric cols:
        X = pd.get_dummies(X, drop_first=True)
        X = sm.add_constant(X, has_constant='add')

        # Numeric cleanup
        X = X.apply(pd.to_numeric, errors='coerce').astype(float)
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        if len(valid_idx) < 5:
            self.error_summary = {"note": "Insufficient rows after cleaning for error testing."}
            self.error_details = None
            return

        Xc = X.loc[valid_idx]
        yc = y.loc[valid_idx]

        # Fit OLS and run simulator
        ols = sm.OLS(yc.values, Xc.values).fit()
        true_beta = ols.params.copy()  # ndarray-like

        sim = RegressionErrorSimulator(ols, Xc, yc, true_beta, alpha=self.alpha)
        sim.fit_and_test()

        self.error_details = sim.type1_type2_error()
        self.error_summary = sim.summary()

    # ----------------------------
    # Public API (compatible)
    # ----------------------------
    def fit_from_dataframe(self, df, target_col, pred_col, sensitive_cols,
                           split_col='split', train_tag='train'):
        """
        Backward-compatible entry point used by your pipeline.
        - Learns global shift on train split
        - Learns relative group offsets on train split
        - Runs Type I/II error testing on train split
        - Returns train and test group tags (shape depends on combine_sensitive)
        """
        # Configure
        self.sensitive_cols = list(sensitive_cols)
        self.audit_log.clear()

        # Split
        train_df = df[df[split_col] == train_tag].copy()
        test_df = df[df[split_col] != train_tag].copy()

        # Outliers (only training)
        train_df = self._remove_outliers(train_df, target_col)

        # Learn global shift
        self.global_shift = self._calc_offset(train_df[target_col], train_df[pred_col])

        # Learn group offsets
        if self.apply_groupwise and len(self.sensitive_cols) > 0:
            self._compute_group_offsets(train_df, target_col, pred_col)
        else:
            self.group_offsets.clear()

        # Type I/II error testing (train split only)
        self._run_error_testing(train_df, target_col)

        # Return tags for train/test in the same shape your pipeline expects
        if self.combine_sensitive:
            return self._generate_tags(train_df), self._generate_tags(test_df)
        else:
            return {c: train_df[c].astype(str) for c in self.sensitive_cols}, \
                   {c: test_df[c].astype(str) for c in self.sensitive_cols}

    def transform(self, y_pred_test, group_test):
        """
        Apply learned corrections to predictions for a test set.
        - y_pred_test: pd.Series aligned to test index
        - group_test: combined tag Series if combine_sensitive=True,
                      else dict of Series keyed by sensitive col
        """
        corrected = y_pred_test.copy()

        # Global component
        if self.apply_global:
            corrected = corrected + self.lambda_strength * self.global_shift

        # Group-wise component
        if self.apply_groupwise and self.group_offsets:
            if self.combine_sensitive:
                # Map combined tags to relative offsets
                tag_to_offset = self.group_offsets.get('__tag__', {})
                tag_series = pd.Series(group_test, index=corrected.index).astype(str)
                rel_offsets = tag_series.map(tag_to_offset).fillna(0.0)
                corrected = corrected + self.lambda_strength * rel_offsets
            else:
                # Sum offsets across sensitive dimensions (if multiple are set)
                total_rel = pd.Series(0.0, index=corrected.index)
                for col in self.sensitive_cols:
                    mapping = self.group_offsets.get(col, {})
                    tags = pd.Series(group_test[col], index=corrected.index).astype(str)
                    total_rel = total_rel + tags.map(mapping).fillna(0.0)
                corrected = corrected + self.lambda_strength * total_rel

        return corrected

    def transform_with_audit(self, y_pred_test_before, group_test):
        """
        Apply transform and produce a simple before/after audit by group means.
        Returns corrected predictions and a dict of audit strings.
        """
        corrected = self.transform(y_pred_test_before.copy(), group_test)
        self.audit_log.clear()

        if self.combine_sensitive:
            tags = pd.Series(group_test, index=corrected.index).astype(str)
            split_cols = tags.str.split('_', expand=True)
            # Ensure we can iterate across sensitive dimensions
            for i, col in enumerate(self.sensitive_cols):
                col_series = split_cols[i]
                values = sorted(col_series.dropna().astype(str).unique())
                lines = [f"Fairness Audit for {col}:"]
                for v in values:
                    mask = (col_series == v).values
                    mean_before = float(y_pred_test_before[mask].mean())
                    mean_after = float(corrected[mask].mean())
                    lines.append(f"- {col}={v}: before={mean_before:.4f}, after={mean_after:.4f}")
                self.audit_log[col] = "\n".join(lines)
        else:
            for col in self.sensitive_cols:
                series = pd.Series(group_test[col], index=corrected.index).astype(str)
                values = sorted(series.dropna().unique())
                lines = [f"Fairness Audit for {col}:"]
                for v in values:
                    mask = (series == v).values
                    mean_before = float(y_pred_test_before[mask].mean())
                    mean_after = float(corrected[mask].mean())
                    lines.append(f"- {col}={v}: before={mean_before:.4f}, after={mean_after:.4f}")
                self.audit_log[col] = "\n".join(lines)

        return corrected, self.audit_log

