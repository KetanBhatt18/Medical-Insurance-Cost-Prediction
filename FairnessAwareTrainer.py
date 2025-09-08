"""
FairnessAwareTrainer.py

Author: [Your Name]
Date: [Current Date]

This module implements FairnessAwareTrainer, a custom regression model trainer
using XGBoost that integrates fairness considerations via a composite loss function.

The composite loss incorporates both the mean and standard deviation of mean absolute error (MAE)
across sensitive groups defined by a chosen feature. This ensures that the model performs not just
accurately overall, but also fairly with reduced disparity between groups.

Users can provide a custom composite loss function or use default additive or multiplicative formulations.

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, DMatrix


class FairnessAwareTrainer(BaseEstimator, RegressorMixin):
    """
    Fairness-aware model trainer wrapping XGBoost regressor with a custom
    objective that regulates both accuracy (mean MAE) and fairness (std deviation MAE across groups).

    The class supports user-defined composite loss functions and employs sample weighting
    to balance imbalance in sensitive groups.

    Parameters
    ----------
    model_params : dict
        Dictionary of parameters for XGBRegressor. Example keys include learning_rate,
        max_depth, subsample, colsample_bytree, tree_method, random_state, etc.
    target : str
        Name of the target variable column in the dataset.
    sensitive_feature : str
        Column name representing the sensitive group feature to audit fairness on (e.g., 'smoker').
    alpha : float, default=1.0
        Weighting factor controlling the importance of the std deviation of group-wise MAE
        in the composite loss function.
    custom_loss_func : callable or None, default=None
        Optional user-defined composite loss function accepting (mean_mae, stddev_mae, alpha)
        and returning a scalar loss value. If None, defaults to multiplicative_composite_loss.
    test_size : float, default=0.2
        Fraction of data reserved for test set during train/test splitting.
    random_state : int, default=42
        Random seed for reproducibility of data splits and model training.
    num_boost_round : int, default=300
        Number of boosting rounds for XGBoost training.
    early_stopping_rounds : int, default=20
        Early stopping patience rounds for XGBoost training.

    Attributes
    ----------
    model_ : xgboost.Booster
        The trained XGBoost model after calling `fit`.
    X_train_ : pd.DataFrame
        Training feature data.
    y_train_ : pd.Series or np.ndarray
        Training target data.
    X_test_ : pd.DataFrame
        Test feature data.
    y_test_ : pd.Series or np.ndarray
        Test target data.
    feature_cols_ : list of str
        List of feature column names used for training and prediction.

    """

    def __init__(self,
                 model_params=None,
                 target='charges',
                 sensitive_feature='smoker',
                 alpha=1.0,
                 custom_loss_func=None,
                 test_size=0.2,
                 random_state=42,
                 num_boost_round=300,
                 early_stopping_rounds=20):
        """
        Initialize the FairnessAwareTrainer with model, data, and training parameters.

        Sets the composite loss function to user-supplied or defaults to multiplicative composite loss.
        """
        # Default parameters if none supplied
        self.model_params = model_params if model_params is not None else {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        self.target = target
        self.sensitive_feature = sensitive_feature

        # Set composite loss function: default to multiplicative if none provided
        if custom_loss_func is None:
            self.custom_loss_func = self.multiplicative_composite_loss
        else:
            self.custom_loss_func = custom_loss_func

        self.alpha = alpha
        self.test_size = test_size
        self.random_state = random_state
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        # Attributes to be set during training
        self.model_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.X_test_ = None
        self.y_test_ = None
        self.feature_cols_ = None

    def _compute_sample_weights(self, groups):
        """
        Compute sample weights inversely proportional to subgroup sizes to mitigate imbalance.

        Parameters
        ----------
        groups : pd.Series or array-like
            Group labels (sensitive feature) for training samples.

        Returns
        -------
        np.ndarray
            Sample weights normalized to mean=1.
        """
        counts = groups.value_counts()
        inv_freq = 1.0 / counts
        weights = groups.map(inv_freq)
        weights /= weights.mean()
        return weights.values

    def _default_composite_loss(self, mean_mae, stddev_mae, alpha):
        """
        Default composite loss function (not used as default, provided for reference):
        additive form Mean MAE + alpha * StdDev MAE.

        Parameters
        ----------
        mean_mae : float
            Mean absolute error across groups.
        stddev_mae : float
            Standard deviation of absolute error across groups.
        alpha : float
            Weighting factor.

        Returns
        -------
        float
            Composite loss scalar.
        """
        return mean_mae + alpha * stddev_mae

    def _custom_obj_closure(self, sample_weights, group_ids, alpha):
        """
        Creates a closure that returns the custom differentiable objective function
        to be used by XGBoost during training.

        This objective function approximates gradient and hessian for the composite loss
        combining mean and std deviation of MAE over sensitive groups, with imbalance
        compensation from sample weights.

        Parameters
        ----------
        sample_weights : np.ndarray
            Sample weights array corresponding to training data.
        group_ids : np.ndarray
            Numeric array indicating group membership of each sample.
        alpha : float
            Weight factor for deviation term in composite loss.

        Returns
        -------
        function
            Custom objective function for XGBoost.
        """
        # Map group IDs to indices (0-based) for indexing internal arrays
        unique_groups = np.unique(group_ids)
        group_id_to_index = {g: i for i, g in enumerate(unique_groups)}

        def custom_obj(y_pred, dtrain):
            """
            Custom objective function called by XGBoost during training.

            Parameters
            ----------
            y_pred : np.ndarray
                Current predictions.
            dtrain : xgboost.DMatrix
                Data matrix containing labels.

            Returns
            -------
            grad : np.ndarray
                Gradient vector.
            hess : np.ndarray
                Hessian vector (second derivatives).
            """
            y_true = dtrain.get_label()
            residual = y_pred - y_true
            abs_residual = np.abs(residual)

            grad = np.zeros_like(y_pred)
            hess = np.ones_like(y_pred)

            # Calculate weighted MAE for each group
            group_maes = [0.0] * len(unique_groups)
            for gid in unique_groups:
                idx = np.where(group_ids == gid)[0]
                if len(idx) > 0:
                    weights = sample_weights[idx]
                    weighted_err = np.average(abs_residual[idx], weights=weights)
                    group_maes[group_id_to_index[gid]] = weighted_err

            mean_mae = np.mean(group_maes)
            std_mae = np.std(group_maes)

            # Compute proxy gradients for each sample
            for gid in unique_groups:
                idx = np.where(group_ids == gid)[0]
                if len(idx) == 0:
                    continue

                # Gradient of MAE approx. with sign of residual
                grad_signs = np.sign(residual[idx])
                # Fairness penalty proxy gradient scaled by alpha
                if std_mae > 1e-8:
                    penalty_grad = alpha * (group_maes[group_id_to_index[gid]] - mean_mae) / std_mae
                else:
                    penalty_grad = 0.0

                # Combine gradients and assign Hessian approximations
                grad[idx] = grad_signs + penalty_grad
                hess[idx] = 1.0

            # Apply sample weights to gradients and Hessians to balance influence
            grad *= sample_weights
            hess *= sample_weights

            return grad, hess

        return custom_obj

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the fairness-aware model on the input data.

        Performs stratified train/test split, computes sample weights to address group imbalance,
        and trains XGBoost model with the custom fairness-aware objective.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data, must include sensitive feature column.
        y : pd.Series
            Target variable.

        Returns
        -------
        self
            Returns self for chaining.
        """
        # Shuffle to randomize order
        X, y = shuffle(X, y, random_state=self.random_state)

        # Stratify split on sensitive feature to preserve group distribution
        stratify_col = X[self.sensitive_feature] if self.sensitive_feature in X.columns else None

        # Split train and test sets
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X, y, test_size=self.test_size, stratify=stratify_col, random_state=self.random_state)

        # Compute sample weights to balance groups in training
        groups_train = self.X_train_[self.sensitive_feature]
        sample_weights = self._compute_sample_weights(groups_train)

        # Map group labels to integer IDs for objective function
        unique_groups = groups_train.unique()
        group_map = {v: i for i, v in enumerate(unique_groups)}
        group_ids = groups_train.map(group_map).values.astype(int)

        # Determine feature columns excluding target
        self.feature_cols_ = [c for c in self.X_train_.columns if c != self.target]

        import xgboost as xgb

        # Prepare DMatrix with sample weights for training and evaluation
        dtrain = xgb.DMatrix(self.X_train_[self.feature_cols_], label=self.y_train_, weight=sample_weights)
        dtest = xgb.DMatrix(self.X_test_[self.feature_cols_], label=self.y_test_)

        # Create custom objective closure with weights, group IDs, and alpha
        custom_obj = self._custom_obj_closure(sample_weights=sample_weights, group_ids=group_ids, alpha=self.alpha)

        # Remove objective key if present; will specify custom obj
        params = self.model_params.copy()
        params.pop('objective', None)
        params['tree_method'] = params.get('tree_method', 'hist')

        # Train model with early stopping
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtest, 'eval')],
            obj=custom_obj,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False, #True,
            #feval=self.r2_eval    # Add this line to evaluate R² during training
        )

        # Make predictions on test set and print composite loss metrics
        y_pred_test = self.model_.predict(dtest)
        mae_mean, mae_std = self._group_mae_stats(self.y_test_, y_pred_test, self.X_test_[self.sensitive_feature])
        composite_loss = self.custom_loss_func(mae_mean, mae_std, self.alpha)

        print(f"Test weighted MAE mean: {mae_mean:.4f}")
        print(f"Test weighted MAE stddev: {mae_std:.4f}")
        print(f"Composite fairness loss: {composite_loss:.4f}")

        return self

    
    def _group_mae_stats(self, y_true, y_pred, groups):
        """
        Compute weighted mean and standard deviation of mean absolute error across groups.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        groups : array-like
            Group labels for each sample.

        Returns
        -------
        tuple of floats
            (weighted_mean_mae, weighted_stddev_mae)
        """
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': groups})
        mae_per_group = df.groupby('group').apply(
            lambda g: mean_absolute_error(g['y_true'], g['y_pred'])
        )
        weights = df['group'].value_counts(normalize=True).loc[mae_per_group.index]
        weighted_mean = np.sum(mae_per_group * weights)
        weighted_std = np.sqrt(np.sum(weights * (mae_per_group - weighted_mean) ** 2))
        return weighted_mean, weighted_std

    
    def predict(self, X: pd.DataFrame):
        """
        Predict target values for new data using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix including all features in training except target.

        Returns
        -------
        np.ndarray
            Array of predicted values.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        dmatrix = DMatrix(X[self.feature_cols_])
        return self.model_.predict(dmatrix)

    
    @staticmethod
    def additive_composite_loss(mean_mae, stddev_mae, alpha):
        """
        Additive composite loss function combining mean and stddev of MAE.

        Parameters
        ----------
        mean_mae : float
            Mean absolute error.
        stddev_mae : float
            Std deviation of absolute error.
        alpha : float
            Weight for stddev term.

        Returns
        -------
        float
            Composite loss value.
        """
        return mean_mae + alpha * stddev_mae

    
    @staticmethod
    def multiplicative_composite_loss(mean_mae, stddev_mae, alpha):
        """
        Multiplicative composite loss function combining mean and stddev of MAE.

        Adds 1 to stddev to avoid multiplication by zero.

        Parameters
        ----------
        mean_mae : float
            Mean absolute error.
        stddev_mae : float
            Std deviation of absolute error.
        alpha : float
            Weight for stddev term.

        Returns
        -------
        float
            Composite loss value.
        """
        return mean_mae * (alpha * (stddev_mae + 1))  # Avoid multiplication by zero


    @staticmethod
    def r2_eval(y_pred, dtrain):
        """
        Custom evaluation metric for XGBoost that returns R² score.
    
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted labels.
        dtrain : xgboost.DMatrix
            Data matrix with true labels.
    
        Returns
        -------
        tuple
            (metric_name, metric_value)
        """
        y_true = dtrain.get_label()
        r2 = r2_score(y_true, y_pred)
        # Return metric name and negative r2 because lower eval metric is better in XGBoost
        # We negate R² since higher is better and XGBoost minimizes eval metrics.
        return 'r2', r2



#------------------------------------------------------------------------------------
