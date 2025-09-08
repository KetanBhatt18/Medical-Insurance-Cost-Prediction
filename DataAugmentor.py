"""
===============================================================================
Data Augmentation Utility Module
===============================================================================
Purpose:
    This Python module provides a reusable `DataAugmentor` class for systematically
    enriching a dataset by varying features within specified ranges and applying
    user-defined transformations and cost functions.

    The augmentation process systematically creates new rows based on modifications
    to existing features, enabling synthetic data generation for model training.

Classes:
    - DataAugmentor:
        * __init__: Initializes the class with a feature configuration dataset
        * _safe_transform: Safely retrieves a transformation function if provided
        * _enrich_one_feature: Generates augmented rows for a single feature
        * augment: Iterates through all configured features to augment the dataset
        * summary: Returns a DataFrame summarizing augmentation activity

===============================================================================
"""

import pandas as pd
import numpy as np
import math



# =============================================================================
# DataAugmentor class
# =============================================================================
class DataAugmentor:
    """
    A class for applying systematic data augmentation based on provided
    feature configurations.
    """



    # -------------------------------------------------------------------------
    # Constructor (__init__)
    # -------------------------------------------------------------------------
    def __init__(self, feature_dataset):
        """
        Initialize the DataAugmentor.

        Parameters:
        ----------
        feature_dataset : pd.DataFrame
            A DataFrame where each row defines an augmentation
            configuration for a feature.
            Expected columns:
                - "feature": str, the feature/column to modify
                - "value_range": tuple(int, int), start and end values for modification
                - "step": int, increment step value
                - "cost_func": callable, function to adjust target column
                - "transform": optional, callable to transform generated values

        Attributes:
        ----------
        feature_dataset : pd.DataFrame
            Stores provided feature configuration.
        history : list of dict
            Keeps a log of augmentation operations performed.
        """
        # Storing the provided configuration dataset
        self.feature_dataset = feature_dataset

        # Initializing history log as empty
        self.history = []



    # -------------------------------------------------------------------------
    # Method: _safe_transform
    # -------------------------------------------------------------------------
    def _safe_transform(self, row):
        """
        Retrieve a transformation function from configuration row.

        Parameters:
        ----------
        row : pd.Series
            A row from feature_dataset containing "transform" key.

        Returns:
        -------
        callable
            The transformation function if valid, else identity function.
        """
        # Extract potential transformation function
        transform_raw = row.get("transform", None)

        # Return valid callable, else identity lambda
        return transform_raw if callable(transform_raw) else (lambda x: x)



    # -------------------------------------------------------------------------
    # Method: _enrich_one_feature
    # -------------------------------------------------------------------------
    def _enrich_one_feature(self, df, row):
        """
        Generate augmented data for a single feature.

        Parameters:
        ----------
        df : pd.DataFrame
            Original dataset to be augmented.
        row : pd.Series
            Feature configuration parameters.

        Returns:
        -------
        pd.DataFrame
            New DataFrame with generated augmented rows.
        """
        # Extract feature details from configuration row
        feature = row["feature"]
        value_range = row["value_range"]
        step = row["step"]
        cost_func = row["cost_func"]

        # Ensure transformation function is safe
        transform = self._safe_transform(row)

        # List to hold enriched dataframes
        enriched = []

        # Counter for number of generated values
        value_count = 0

        # Print task start message
        print(f"\n- Augmenting feature '{feature}' with values from {value_range[0]} to {value_range} (step {step})...", end=" ")

        # Iterate over defined value range in steps
        for val in range(*value_range, step):
            # Apply transformation to current value
            val_transformed = transform(val)

            # Create deep copy of original dataframe
            df_copy = df.copy()

            # Set current feature to transformed value
            df_copy[feature] = val_transformed

            # Apply cost adjustment function to target column 'charges'
            df_copy["charges"] = df_copy.apply(
                lambda r: r["charges"] + cost_func(r[feature]),
                axis=1
            )

            # Append modified dataframe to list
            enriched.append(df_copy)

            # Increment generated value counter
            value_count += 1

        # Concatenate all enriched dataframes
        df_result = pd.concat(enriched, ignore_index=True)

        # Track detailed summary for later inspection
        self.history.append({
            "feature": feature,
            "value_count": value_count,
            "pre_shape": df.shape[0],
            "post_shape": df_result.shape
        })

        # Print completion message
        print("Dataset Shape:", df_result.shape)

        return df_result



    # -------------------------------------------------------------------------
    # Method: augment
    # -------------------------------------------------------------------------
    def augment(self, df):
        """
        Augment dataset using all configured feature transformations.

        Parameters:
        ----------
        df : pd.DataFrame
            The input dataset that will be augmented.

        Returns:
        -------
        pd.DataFrame
            Fully augmented dataset including new synthetic rows.
        """
        # Print starting augmentation process
        print("- Starting dataset augmentation...", end=" ")

        # Copy original dataset to avoid mutation
        df_aug = df.copy()

        # Reset history before every augmentation run
        self.history = []

        # Iterate through each feature configuration
        for _, row in self.feature_dataset.iterrows():
            # Enrich dataset for the current feature
            df_aug = self._enrich_one_feature(df_aug, row)

        # Print completion message
        print("done.")

        return df_aug



    # -------------------------------------------------------------------------
    # Method: summary
    # -------------------------------------------------------------------------
    def summary(self):
        """
        Return a summary of augmentation operations.

        Returns:
        -------
        pd.DataFrame
            DataFrame summarizing number of generated values, feature names,
            and changes in dataset shape post augmentation.
        """
        # Converting stored history into a DataFrame summary
        return pd.DataFrame(self.history)
