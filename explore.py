"""
===============================================================================
Data Exploration and Visualization Utilities
===============================================================================
Purpose:
    Provides a comprehensive suite of helper functions for exploratory data analysis,
    including descriptive statistics, data validation, encoding, duplicate inspection,
    and various visualization utilities using matplotlib and seaborn.

Functions:
    - describe_column_stats: Summarize column types, unique values, sample values, min/max.
    - read_and_verify_dataset: Load and validate datasets, display summary.
    - show_duplicates_in_pairs: Display duplicate records in paired format.
    - describe_with_features: Enhanced descriptive stats (CV, skew, IQR, outlier spread, etc).
    - encode_features: Feature encoding (binary, one-hot).
    - univariate_plots: Visualize distributions and boxplots for all columns.
    - univariate_plots_with_hue: Univariate plots stratified by hue column.
    - bivariate_plots: Smart bivariate plot (scatter/box/count) for target vs features.
    - bivariate_plots_with_hue: Bivariate plots with grouping/hue.
    - create_upper_triangle_mask: Get upper-triangle mask for correlation heatmaps.
    - plot_correlation_heatmap: Visualize variable correlations as heatmap.
    - plot_scatter_matrix: Multi-plot scatter chart given column pairs.
    - plot_scatter_matrix_hue: Multi-plot scatter with hue/subgroup overlays.
    - xy_scatter: Quickly plot target vs all features.
    - xy_scatter_with_hue: Same with hue overlays.
    - eda: End-to-end exploratory analysis (stats, plots, correlations).
===============================================================================
"""

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Save object
import joblib

# Supress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)





# =============================================================================
# Function: describe_column_stats
# =============================================================================

def describe_column_stats(df, max_sample_display=5, max_unique_for_sampling=25):
    """
    Summarize type, unique count, range, and sample values for each DataFrame column.
    """
    print("- Generating dataframe column summary...", end=" ")
    summary = []
    
    for col in df.columns:
        col_data = df[col]
        col_type = col_data.dtype

        # Get all unique non-null values
        unique_vals = pd.Series(col_data.dropna().unique())
        try:
            sorted_vals = np.sort(unique_vals)
        except Exception:
            sorted_vals = unique_vals.sort_values()

        unique_count = len(sorted_vals)

        # -------- UPDATED: Min and Max from sorted list --------
        if unique_count > 0:
            min_val = sorted_vals[0]
            max_val = sorted_vals[-1]
        else:
            min_val = None
            max_val = None

        # -------- UPDATED: Limit decimal places for floats --------
        def format_val(val):
            if isinstance(val, float):
                return round(val, 2)  # Round to 2 decimals
            return val

        min_val = format_val(min_val)
        max_val = format_val(max_val)

        # Show sample values for categorical/low-unique
        if unique_count <= max_unique_for_sampling:
            sample_list = [format_val(v) for v in sorted_vals[:max_sample_display]]
            sample_vals = ', '.join(map(str, sample_list))
            if unique_count > max_sample_display:
                sample_vals += "..."
        elif unique_count > 10:
            first_five = ', '.join(map(str, [format_val(v) for v in sorted_vals[:5]]))
            last_five  = ', '.join(map(str, [format_val(v) for v in sorted_vals[-5:]]))
            sample_vals = f"{first_five} ... {last_five}"
        else:
            sample_vals = ""

        summary.append({
            'Column': col,
            'Type': str(col_type),
            'Unique Counts': unique_count,
            'Min': min_val,
            'Max': max_val,
            'Unique Sample Values': sample_vals
        })
    
    print("done.")
    return pd.DataFrame(summary)






# =============================================================================
# Function: read_and_verify_dataset
# =============================================================================
def read_and_verify_dataset(filename, col=None, sep=',', index_col=None, header='infer'):
    """
    Loads a dataset and prints basic metadata and summary using describe_column_stats.

    Parameters:
    ----------
    filename: str
        File path.
    col: list or None
        Column names, optional.
    sep: str
        CSV delimiter.
    index_col: int or str or None
        Index column.
    header: 'infer', int or None
        Row(s) to use as column names.

    Returns:
    -------
    tuple: (pd.DataFrame, pd.DataFrame)
        Loaded DataFrame, summary DataFrame.
    """
    from IPython.display import display

    print(f"- Reading dataset from {filename}...", end=" ")
    try:
        df = pd.read_csv(filename, names=col, sep=sep, index_col=index_col, header=header)
        print("done.")

        print("- Displaying first and last 5 rows:")
        display(df)

        print("\n- Displaying dataset info:")
        display(df.info())

        print(f"\n- Missing values: {df.isna().sum().sum()}")
        print(f"- Duplicate rows: {df.duplicated().sum()}")

        print("\n- Column-wise summary:")
        summary_df = describe_column_stats(df)
        display(summary_df)
        return df, summary_df
    except Exception as e:
        print(f"❌ Error reading or verifying the file: {e}")
        return None, None





# =============================================================================
# Function: show_duplicates_in_pairs
# =============================================================================
def show_duplicates_in_pairs(df, subset=None, keep=False):
    """
    Pairwise display of duplicate records for data cleaning.

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame to inspect.
    subset: list, optional
        Columns to consider for duplicate detection.
    keep: bool or str
        Whether to keep first/last/False.

    Returns:
    -------
    pd.DataFrame
        Duplicates paired side by side.
    """
    print("- Identifying and grouping duplicate rows...", end=" ")

    mask = df.duplicated(subset=subset, keep=keep)
    dup_df = df[mask].copy()
    dup_df['__orig_index__'] = dup_df.index
    dup_df.reset_index(drop=True, inplace=True)

    # Pad if odd number of duplicates
    if len(dup_df) % 2 != 0:
        pad = pd.DataFrame([[np.nan] * dup_df.shape[1]], columns=dup_df.columns)
        dup_df = pd.concat([dup_df, pad], ignore_index=True)
    mid = len(dup_df) // 2
    left = dup_df.iloc[:mid].reset_index(drop=True)
    right = dup_df.iloc[mid:].reset_index(drop=True)

    left_cols = [f"{col}_A" for col in df.columns] + ['Index_A']
    right_cols = [f"{col}_B" for col in df.columns] + ['Index_B']

    paired_df = pd.concat([left, right], axis=1)
    paired_df.columns = left_cols + right_cols

    print("done.")
    return paired_df





# =============================================================================
# Function: describe_with_features
# =============================================================================
def describe_with_features(df):
    """
    Compute extended descriptive statistics for numeric columns.

    Parameters:
    ----------
    df: pd.DataFrame

    Returns:
    -------
    pd.DataFrame
        Transposed DataFrame with extra features.
    """
    import pandas as pd
    print("- Calculating enhanced descriptive statistics...", end=" ")
    try:
        desc = df.describe(include='all').T.copy()
        desc["cv"] = desc["std"] / desc["mean"]
        desc["IQR"] = desc["75%"] - desc["25%"]
        desc["skew"] = 3 * (desc["mean"] - desc["50%"]) / desc["std"]
        desc["upper_lim"] = desc["50%"] + 1.5 * (desc["75%"] - desc["50%"])
        desc["lower_lim"] = desc["50%"] - 1.5 * (desc["50%"] - desc["25%"])
        desc["outlier_spread"] = (desc["max"] - desc["min"]) / (desc["upper_lim"] - desc["lower_lim"])
        print("done.")
        return desc.T.round(3)
    except Exception as err:
        print(f"Descriptive stats error: {err}")
        return pd.DataFrame()





# =============================================================================
# Function: ReplaceEncode
# =============================================================================
def ReplaceEncode(df, replace_dict=None, one_hot_features=None, drop_first=True):
    """
    Generalized encoding function for categorical features:
    - Performs replacements based on provided dictionary.
    - One-hot encodes specified features.
    - One function for label & one-hot-encoding needs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    replace_dict : dict, optional
        Dictionary mapping feature names to replacement dictionaries,
        e.g. {'sex': {'male': 1, 'female': 0}, 'smoker': {'yes': 1, 'no': 0}}
    one_hot_features : list of str, optional
        List of features to one-hot encode. If None, all object/categorical columns
        except those in replace_dict are encoded.
    drop_first : bool, default True
        Whether to drop the first level in one-hot encoding.

    Returns
    -------
    pd.DataFrame
        Encoded DataFrame.
    """
    df_encoded = df.copy()
    print("🔄 Starting encoding process...")

    # 1. Apply replacement maps if given
    if replace_dict is not None:
        print("🔧 Applying label replacements...")
        for feature, mapping in replace_dict.items():
            if feature in df_encoded.columns:
                df_encoded[feature] = df_encoded[feature].replace(mapping)
                print(f"✅ Feature: '{feature}': {mapping} : Done")
            else:
                print(f"⚠️ Feature: '{feature}' not found in DataFrame. Skipping replacement.")

    # 2. Identify features to one-hot encode
    if one_hot_features is None:
        print("🔍 Auto-detecting categorical features for one-hot encoding...")
        replace_keys = set(replace_dict.keys()) if replace_dict else set()
        one_hot_features = [
            col for col in df_encoded.select_dtypes(include=['object', 'category']).columns
            if col not in replace_keys
        ]
        print(f"📌 Features selected for one-hot encoding: {one_hot_features}")

    # 3. Apply one-hot encoding
    if one_hot_features:
        print("🎯 Applying one-hot encoding...")
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=one_hot_features,
            drop_first=drop_first,
            dtype=int
        )
        encoded_cols = [col for col in df_encoded.columns if any(f"{feat}_" in col for feat in one_hot_features)]
        print(f"✅ One-hot encoded features: {encoded_cols}")
    else:
        print("ℹ️ No features selected for one-hot encoding.")

    print("🏁 Encoding process completed. Please find below Encoded Dataset:")
    display(df_encoded)
    return df_encoded







# =============================================================================
# Function: univariate_plots
# =============================================================================
def univariate_plots(df, columns=4):
    """
    Plot KDE/countplot and boxplot for every attribute.

    Parameters:
    ----------
    df: pd.DataFrame
    columns: int
        Number of subplot columns.

    Returns: None
    """
    print("- Generating univariate plots...", end=" ")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_features = len(df.columns)
    rows = int(np.ceil(num_features / columns))
    fig, axes = plt.subplots(rows * 2, columns, figsize=(columns * 4, rows * 5))
    axes = np.array(axes).reshape(rows * 2, columns)

    for idx, feature in enumerate(df.columns):
        r = (idx // columns) * 2
        c = idx % columns
        ax_dist = axes[r, c]
        ax_box = axes[r + 1, c]
        x_min, x_max = df[feature].min(), df[feature].max()
        x_range = (x_min, x_max)
        if df[feature].nunique() > 20 and pd.api.types.is_numeric_dtype(df[feature]):
            sns.kdeplot(df[feature], ax=ax_dist, fill=True, color='steelblue')
            ax_dist.axvline(df[feature].mean(), color="green", linestyle="--", label="Mean")
            ax_dist.axvline(df[feature].median(), color="black", linestyle="-", label="Median")
            ax_dist.set_xlim(x_range)
            ax_dist.legend()
        else:
            cp = sns.countplot(x=df[feature], ax=ax_dist, color="darkgreen")
            cp.bar_label(cp.containers[0])
        sns.boxplot(x=df[feature], ax=ax_box, color="orange", showmeans=True)
        ax_box.set_xlim(x_range)
        ax_dist.set_title(f"Distribution: {feature}", fontsize=10)
        ax_box.set_title(f"Boxplot: {feature}", fontsize=10)
    total_rows = rows * 2
    total_slots = total_rows * columns
    used_slots = num_features * 2
    for j in range(used_slots, total_slots):
        fig.delaxes(axes.flatten()[j])
    plt.tight_layout()
    plt.show()
    print("done.")





# =============================================================================
# Function: univariate_plots_with_hue
# =============================================================================
def univariate_plots_with_hue(df, hue, columns=4):
    """
    Plot KDE/countplot and boxplot for every attribute, with stratification by hue.

    Parameters:
    ----------
    df: pd.DataFrame
    hue: str
        Column for grouping.
    columns: int

    Returns: None
    """
    print(f"- Generating univariate plots with hue='{hue}'...", end=" ")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = [col for col in df.columns if col != hue]
    rows = int(np.ceil(len(features) / columns))
    fig, axes = plt.subplots(rows * 2, columns, figsize=(columns * 4, rows * 5))
    axes = np.array(axes).reshape(rows * 2, columns)
    for idx, feature in enumerate(features):
        r = (idx // columns) * 2
        c = idx % columns
        ax_dist = axes[r, c]
        ax_box = axes[r + 1, c]
        if df[feature].nunique() > 20 and pd.api.types.is_numeric_dtype(df[feature]):
            sns.kdeplot(data=df, x=feature, hue=hue, ax=ax_dist, fill=True, warn_singular=False)
            ax_dist.axvline(df[feature].mean(), color="green", linestyle="--", label="Mean")
            ax_dist.axvline(df[feature].median(), color="black", linestyle="-", label="Median")
            ax_dist.legend()
        else:
            count_data = df.groupby([feature, hue]).size().reset_index(name='count')
            pivot = count_data.pivot(index=feature, columns=hue, values='count').fillna(0)
            pivot.plot(kind='bar', stacked=True, ax=ax_dist, colormap="Set3")
        sns.boxplot(x=df[feature], y=df[hue], ax=ax_box, palette="Set2", showmeans=True, orient='h')
        ax_dist.set_title(f"Distribution: {feature}", fontsize=10)
        ax_box.set_title(f"Boxplot: {feature} by {hue}", fontsize=10)
        ax_dist.tick_params(axis='x', labelrotation=45)
    for j in range(len(features) * 2, rows * 2 * columns):
        fig.delaxes(axes.flatten()[j])
    plt.tight_layout()
    plt.show()
    print("done.")





# =============================================================================
# Function: bivariate_plots
# =============================================================================
def bivariate_plots(df, target, columns=4):
    """
    Plot scatter/boxplot/count chart for each feature vs. target.

    Parameters:
    ----------
    df: pd.DataFrame
    target: str
        Column to use as y-axis.
    columns: int

    Returns: None
    """
    print(f"- Creating bivariate plots for '{target}'...", end=" ")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = [col for col in df.columns if col != target]
    num_features = len(features)
    rows = int(np.ceil(num_features / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 4))
    axes = np.array(axes).reshape(rows, columns)
    for idx, feature in enumerate(features):
        r, c = divmod(idx, columns)
        ax = axes[r, c]
        x = df[feature]
        y = df[target]
        is_x_cat = not pd.api.types.is_numeric_dtype(x) or x.nunique() <= 20
        is_y_cat = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20
        if not is_x_cat and not is_y_cat:
            sns.scatterplot(x=x, y=y, ax=ax, color="dodgerblue", alpha=0.6)
            ax.set_title(f"{target} vs {feature} (Scatter)", fontsize=10)
        elif is_x_cat and not is_y_cat:
            sns.boxplot(x=x, y=y, ax=ax, palette="Set2", showmeans=True)
            ax.set_title(f"{target} by {feature} (Boxplot)", fontsize=10)
        elif not is_x_cat and is_y_cat:
            sns.boxplot(y=y, x=x, ax=ax, palette="Set3", showmeans=True, orient='h')
            ax.set_title(f"{feature} by {target} (Boxplot)", fontsize=10)
        else:
            stacked = df.groupby([feature, y.name]).size().reset_index(name='count')
            stacked_pivot = stacked.pivot(index=feature, columns=y.name, values='count').fillna(0)
            stacked_pivot.plot(kind='bar', stacked=True, ax=ax, colormap="viridis")
            ax.set_title(f"{target} by {feature} (Countplot)", fontsize=10)
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")
            ax.legend(title=target)
        ax.tick_params(axis='x', labelrotation=45)
    for j in range(num_features, rows * columns):
        fig.delaxes(axes.flatten()[j])
    plt.tight_layout()
    plt.show()
    print("done.")





# =============================================================================
# Function: bivariate_plots_with_hue
# =============================================================================
def bivariate_plots_with_hue(df, target, hue, columns=4):
    """
    Plot bivariate chart for each feature vs. target with hue overlays.

    Parameters:
    ----------
    df: pd.DataFrame
    target: str
        Main comparison y-axis column.
    hue: str
        Subgroup column.
    columns: int

    Returns: None
    """
    print(f"- Creating bivariate plots for '{target}' with hue '{hue}'...", end=" ")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = [col for col in df.columns if col not in [target, hue]]
    num_features = len(features)
    rows = int(np.ceil(num_features / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 4))
    axes = np.array(axes).reshape(rows, columns)
    for idx, feature in enumerate(features):
        r, c = divmod(idx, columns)
        ax = axes[r, c]
        x = df[feature]
        y = df[target]
        is_x_cat = not pd.api.types.is_numeric_dtype(x) or x.nunique() <= 20
        is_y_cat = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20
        if not is_x_cat and not is_y_cat:
            sns.scatterplot(data=df, x=feature, y=target, hue=hue, ax=ax, alpha=0.6)
            ax.set_title(f"{target} vs {feature} (Scatter)", fontsize=10)
        elif is_x_cat and not is_y_cat:
            sns.boxplot(data=df, x=feature, y=target, hue=hue, ax=ax, palette="Set2", showmeans=True)
            ax.set_title(f"{target} by {feature} (Box)", fontsize=10)
        elif not is_x_cat and is_y_cat:
            sns.boxplot(data=df, y=target, x=feature, hue=hue, ax=ax, palette="Set3", showmeans=True, orient='h')
            ax.set_title(f"{feature} by {target} (Box)", fontsize=10)
        else:
            count_data = df.groupby([feature, target, hue]).size().reset_index(name='count')
            pivot = count_data.pivot_table(index=feature, columns=[target, hue], values='count', fill_value=0)
            pivot.groupby(level=0, axis=1).sum().plot(kind='bar', stacked=True, ax=ax, colormap="viridis")
            ax.set_title(f"{target} by {feature} (Stacked)", fontsize=10)
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")
        ax.tick_params(axis='x', labelrotation=45)
    for j in range(num_features, rows * columns):
        fig.delaxes(axes.flatten()[j])
    plt.tight_layout()
    plt.show()
    print("done.")





# =================
# Function: create_upper_triangle_mask
# =================
def create_upper_triangle_mask(ncols):
    """
    Return boolean mask array for upper triangle of correlation matrix.
    Parameters:
    ----------
    ncols: int
    Returns:
    -------
    np.ndarray
        Boolean mask.
    """
    import numpy as np
    # Create mask for upper triangle
    return np.triu(np.ones((ncols, ncols), dtype=bool))


# =================
# Function: plot_correlation_heatmap
# =================
def plot_correlation_heatmap(df, method='spearman'):
    """
    Plot heatmap of correlation matrix with upper triangle masked.
    Parameters:
    ----------
    df: pd.DataFrame
    method: str
        Correlation method: 'pearson', 'spearman', 'kendall'.
    Returns: None
    """
    print(f"- Calculating correlation matrix using '{method}'...", end=" ")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.shape[1] < 2:
        print("Correlation heatmap requires at least two columns.")
        return

    try:
        # Compute correlation matrix
        corr_matrix = df.corr(method=method).round(2)

        # Get number of columns (or rows, since square matrix)
        ncols = corr_matrix.shape[1]

        # Create upper triangle mask
        mask = create_upper_triangle_mask(ncols)

        # Plot heatmap
        plt.figure(figsize=(ncols * 1.0, ncols * 0.6))
        sns.heatmap(
            corr_matrix,
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            annot=True,
            annot_kws={"size": 10},
            mask=mask,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7}
        )

        plt.title(f"{method.capitalize()} Correlation Heatmap", fontsize=14, pad=10)
        plt.tight_layout()
        plt.show()
        print("done.")

    except Exception as e:
        print(f"⚠️ Error generating correlation heatmap: {e}")


# =============================================================================
# Function: plot_scatter_matrix
# =============================================================================
def plot_scatter_matrix(df, feature_pairs, columns=4):
    """
    Plot grid of (x,y) scatterplots for supplied column pairs.

    Parameters:
    ----------
    df: pd.DataFrame
    feature_pairs: list of (str, str)
        List of (x,y) feature name pairs.
    columns: int

    Returns: None
    """
    print("- Plotting scatter matrix for feature pairs...", end=" ")

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_plots = len(feature_pairs)
    rows = int(np.ceil(num_plots / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 4))
    axes = axes.flatten()
    for idx, (x_col, y_col) in enumerate(feature_pairs):
        ax = axes[idx]
        if x_col not in df.columns or y_col not in df.columns:
            ax.set_title(f"⚠️ Missing: {x_col} vs {y_col}", fontsize=10)
            ax.axis("off")
            continue
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=10)
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    print("done.")





# =============================================================================
# Function: plot_scatter_matrix_hue
# =============================================================================
def plot_scatter_matrix_hue(df, feature_pairs, hue=None, columns=4):
    """
    Plot grid of (x,y) scatterplots with hue grouping.

    Parameters:
    ----------
    df: pd.DataFrame
    feature_pairs: list of (str, str)
        List of feature (x,y) pairs.
    hue: str, optional
        Hue column for coloring.
    columns: int

    Returns: None
    """
    print(f"- Plotting scatter matrix with hue ({hue})...", end=" ")

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_plots = len(feature_pairs)
    rows = int(np.ceil(num_plots / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.5))
    axes = axes.flatten()
    for idx, (x_col, y_col) in enumerate(feature_pairs):
        ax = axes[idx]
        if x_col not in df.columns or y_col not in df.columns:
            ax.set_title(f"⚠️ Missing: {x_col} vs {y_col}", fontsize=10)
            ax.axis("off")
            continue
        plot_kwargs = {}
        if hue and hue in df.columns:
            plot_kwargs['hue'] = df[hue]
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, **plot_kwargs)
        ax.set_title(f"{x_col} vs {y_col}" + (f" by {hue}" if hue else ""), fontsize=10)
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    print("done.")





# =============================================================================
# Function: xy_scatter
# =============================================================================
def xy_scatter(data, y, cols_num=4):
    """
    Plot scatter of target vs every feature.

    Parameters:
    ----------
    data: pd.DataFrame
    y: str
        Name of target var.
    cols_num: int

    Returns: None
    """
    print(f"- Plotting {y} against all other features...", end=" ")
    try:
        if y not in data.columns:
            raise ValueError(f"Target column '{y}' not found in the dataset.")
        feature_pairs = [[col, y] for col in data.columns if col != y]
        plot_scatter_matrix(data, feature_pairs=feature_pairs, columns=cols_num)
        print("done.")
    except Exception as e:
        print(f"⚠️ Error creating scatter plots: {e}")





# =============================================================================
# Function: xy_scatter_with_hue
# =============================================================================
def xy_scatter_with_hue(data, y, hue=None, cols_num=4):
    """
    Plot scatter target vs all features, with optional hue coloring.

    Parameters:
    ----------
    data: pd.DataFrame
    y: str
        Target variable.
    hue: str, optional
        Hue/category/subgroup column.
    cols_num: int

    Returns: None
    """
    print(f"- Plotting {y} vs all features", end=" ")
    if hue:
        print(f"with hue '{hue}'...", end=" ")
    else:
        print("...", end=" ")
    try:
        if y not in data.columns:
            raise ValueError(f"Target column '{y}' not found in the dataset.")
        if hue and hue not in data.columns:
            raise ValueError(f"Hue column '{hue}' not found in the dataset.")
        feature_pairs = [[col, y] for col in data.columns if col not in [y, hue]]
        if hue:
            plot_scatter_matrix_hue(data, feature_pairs=feature_pairs, hue=hue, columns=cols_num)
        else:
            plot_scatter_matrix(data, feature_pairs=feature_pairs, columns=cols_num)
        print("done.")
    except Exception as e:
        print(f"⚠️ Error creating scatter plots with hue: {e}")





# =============================================================================
# Function: eda
# =============================================================================
def eda(df, y_variable, hue=None, correlation_method='spearman', univariate_cols=4):
    """
    Perform a complete exploratory data analysis (EDA) workflow.

    Parameters:
    ----------
    df: pd.DataFrame
    y_variable: str
        Target column name.
    hue: str, optional
        Column for group overlays (optional).
    correlation_method: str

    Returns: None
    """
    from IPython.display import display

    print("\n- Descriptive Statistics with Enhanced Features:")
    display(describe_with_features(df))

    print("\n- Univariate Distribution & Boxplots:")
    if hue:
        univariate_plots_with_hue(df, hue)
    else:
        univariate_plots(df)

    print("\n- Correlation Heatmap:")
    plot_correlation_heatmap(df.select_dtypes(include=np.number), method=correlation_method)

    print(f"\n- Bi-variate plot of attributes with '{y_variable}':")
    if hue:
        bivariate_plots_with_hue(df, y_variable, hue)
    else:
        bivariate_plots(df, y_variable)
    print("done.")
