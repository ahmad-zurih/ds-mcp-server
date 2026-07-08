"""
Statistical Analysis Module for the AI Visualization Engine.
Provides tools for the LLM to run advanced statistical tests using Pingouin and Statsmodels.
"""

import pandas as pd
import pingouin as pg
import statsmodels.api as sm

from .viz_utils import load_data_safely

# Values that represent the "positive" class in binary categorical columns.
# Used by rank_target_correlations for deterministic encoding.
_KNOWN_POSITIVE_VALUES = {
    "m", "malignant", "yes", "true", "1", "positive", "pos",
    "disease", "infected", "bad", "high", "abnormal"
}


def _stats_code_header(data_file_path: str) -> str:
    """Returns the import/load preamble for a reproducible stats snippet."""
    ext = data_file_path.rsplit(".", 1)[-1].lower()
    if ext == "tsv":
        loader = "df = pd.read_csv('your_data.tsv', sep='\\t')"
    elif ext in ("xls", "xlsx"):
        loader = f"df = pd.read_excel('your_data.{ext}')"
    elif ext == "json":
        loader = "df = pd.read_json('your_data.json')"
    else:
        loader = "df = pd.read_csv('your_data.csv')"
    return f"import pandas as pd\n\n# Load Data\n{loader}\n\n"


def run_correlation_impl(
    data_file_path: str, x_column: str, y_column: str, method: str = "pearson"
) -> str:
    """
    Computes the correlation between two numeric columns.

    Args:
        data_file_path: The absolute path to the data file.
        x_column: The name of the first numeric column.
        y_column: The name of the second numeric column.
        method: The correlation method ('pearson', 'spearman', or 'kendall').

    Returns:
        A markdown-formatted string of the correlation results + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' and/or '{y_column}' not found."

        clean_df = df[[x_column, y_column]].dropna()
        if len(clean_df) < 3:
            return "Error: Not enough valid data points to calculate correlation."

        res = pg.corr(clean_df[x_column], clean_df[y_column], method=method)

        code = (
            _stats_code_header(data_file_path)
            + "import pingouin as pg\n\n"
            + f"clean_df = df[['{x_column}', '{y_column}']].dropna()\n"
            + f"result = pg.corr(clean_df['{x_column}'], clean_df['{y_column}'], method='{method}')\n"
            + "print(result)"
        )
        return (
            f"Correlation Analysis ({method}) between '{x_column}' and '{y_column}':\n\n"
            f"{res.to_markdown()}\n\n"
            f"```python\n{code}\n```"
        )
    except Exception as e:
        return f"Error computing correlation: {str(e)}"


def run_group_comparison_impl(
    data_file_path: str, target_col: str, group_col: str
) -> str:
    """
    Automatically performs a T-test (for 2 groups) or ANOVA (for >2 groups) 
    on a target numeric variable grouped by a categorical variable.

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The numeric column to test.
        group_col: The categorical column defining the groups.

    Returns:
        A markdown-formatted string of the test results + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if target_col not in df.columns or group_col not in df.columns:
            return f"Error: Columns '{target_col}' or '{group_col}' not found."

        clean_df = df[[target_col, group_col]].dropna()
        groups = sorted(clean_df[group_col].unique(), key=str)

        if len(groups) == 2:
            group1 = clean_df[clean_df[group_col] == groups[0]][target_col]
            group2 = clean_df[clean_df[group_col] == groups[1]][target_col]
            res = pg.ttest(group1, group2)
            result_text = (
                f"Independent T-test for '{target_col}' grouped by '{group_col}' "
                f"(Groups: {groups[0]} vs {groups[1]}):\n\n{res.to_markdown()}"
            )
            code_logic = (
                f"clean_df = df[['{target_col}', '{group_col}']].dropna()\n"
                f"g1 = clean_df[clean_df['{group_col}'] == '{groups[0]}']['{target_col}']\n"
                f"g2 = clean_df[clean_df['{group_col}'] == '{groups[1]}']['{target_col}']\n"
                "result = pg.ttest(g1, g2)\nprint(result)"
            )
        elif len(groups) > 2:
            res = pg.anova(dv=target_col, between=group_col, data=clean_df, detailed=True)
            result_text = (
                f"One-way ANOVA for '{target_col}' grouped by '{group_col}' "
                f"({len(groups)} groups):\n\n{res.to_markdown()}"
            )
            code_logic = (
                f"clean_df = df[['{target_col}', '{group_col}']].dropna()\n"
                f"result = pg.anova(dv='{target_col}', between='{group_col}', data=clean_df, detailed=True)\n"
                "print(result)"
            )
        else:
            return "Error: The grouping column must have at least 2 unique values."

        code = (
            _stats_code_header(data_file_path)
            + "import pingouin as pg\n\n"
            + code_logic
        )
        return f"{result_text}\n\n```python\n{code}\n```"

    except Exception as e:
        return f"Error computing group comparison: {str(e)}"


def run_linear_regression_impl(
    data_file_path: str, target_col: str, predictor_cols: list[str]
) -> str:
    """
    Performs an Ordinary Least Squares (OLS) linear regression.

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The dependent variable (Y).
        predictor_cols: A list of independent variables (X).

    Returns:
        A formatted string of the statsmodels regression summary + reproducible code,
        or an error message.
    """
    try:
        df = load_data_safely(data_file_path)

        # Coerce predictor_cols to a list in case the LLM passed a comma-separated string.
        if isinstance(predictor_cols, str):
            predictor_cols = [c.strip() for c in predictor_cols.split(",") if c.strip()]

        missing_cols = [col for col in [target_col] + predictor_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing columns in data: {', '.join(missing_cols)}"

        cols = [target_col] + predictor_cols
        clean_df = df[cols].dropna()
        
        if len(clean_df) <= len(predictor_cols):
            return "Error: Not enough data points to run regression after dropping NaNs."

        X = sm.add_constant(clean_df[predictor_cols])
        y = clean_df[target_col]
        model = sm.OLS(y, X).fit()

        pred_list = str(predictor_cols)
        code = (
            _stats_code_header(data_file_path)
            + "import statsmodels.api as sm\n\n"
            + f"cols = ['{target_col}'] + {pred_list}\n"
            + "clean_df = df[cols].dropna()\n"
            + f"X = sm.add_constant(clean_df[{pred_list}])\n"
            + f"y = clean_df['{target_col}']\n"
            + "model = sm.OLS(y, X).fit()\n"
            + "print(model.summary())"
        )
        return (
            f"OLS Linear Regression Results (Target: {target_col}):\n\n"
            f"{model.summary().as_text()}\n\n"
            f"```python\n{code}\n```"
        )
    except Exception as e:
        return f"Error computing linear regression: {str(e)}"


def rank_target_correlations_impl(
    data_file_path: str, target_col: str, method: str = "pearson"
) -> str:
    """
    Calculates the correlation between a target column and all other numeric features,
    sorting them by absolute correlation strength. Automatically encodes binary text
    deterministically (known positive class → 1, otherwise alphabetically lower → 0).

    Args:
        data_file_path: The absolute path to the data file.
        target_col: The column to correlate all other features against.
        method: The correlation method ('pearson' or 'spearman').

    Returns:
        A markdown table ranking the features + reproducible code, or an error message.
    """
    try:
        df = load_data_safely(data_file_path)
        if target_col not in df.columns:
            return f"Error: Target column '{target_col}' not found in the dataset."

        working_df = df.copy()
        binary_encode_snippet = ""

        if not pd.api.types.is_numeric_dtype(working_df[target_col]):
            raw_unique = working_df[target_col].dropna().unique()
            if len(raw_unique) != 2:
                return (
                    f"Error: Target column '{target_col}' is non-numeric and contains "
                    f"{len(raw_unique)} unique values. It must be strictly binary to auto-encode."
                )
            # Sort so the result is deterministic regardless of row order.
            sorted_vals = sorted(raw_unique, key=lambda x: str(x).lower())
            # Check whether either value is a known "positive" class.
            pos_idx = next(
                (i for i, v in enumerate(sorted_vals) if str(v).lower() in _KNOWN_POSITIVE_VALUES),
                1  # default: alphabetically second value → 1
            )
            neg_idx = 1 - pos_idx
            val_map = {sorted_vals[pos_idx]: 1, sorted_vals[neg_idx]: 0}
            working_df[target_col] = working_df[target_col].map(val_map)
            # Build the encoding step to include in the reproducible snippet.
            binary_encode_snippet = (
                f"# Binary encode target column\n"
                f"val_map = {{{repr(sorted_vals[pos_idx])}: 1, {repr(sorted_vals[neg_idx])}: 0}}\n"
                f"df['{target_col}'] = df['{target_col}'].map(val_map)\n\n"
            )

        numeric_df = working_df.select_dtypes(include=["number"])
        if target_col not in numeric_df.columns:
            return f"Error: Target column '{target_col}' could not be evaluated numerically."

        correlations = numeric_df.corr(method=method)[target_col].drop(target_col)

        if correlations.empty:
            return "Error: No other numeric columns found to correlate against."

        ranking_df = pd.DataFrame({
            "Feature": correlations.index,
            "Correlation Coefficient": correlations.values,
            "Absolute Strength": correlations.abs().values
        }).sort_values(by="Absolute Strength", ascending=False).drop(columns=["Absolute Strength"])

        code = (
            _stats_code_header(data_file_path)
            + binary_encode_snippet
            + f"corr = df.select_dtypes(include='number').corr(method='{method}')['{target_col}']\n"
            + f"corr = corr.drop('{target_col}').abs().sort_values(ascending=False)\n"
            + "print(corr)"
        )
        return (
            f"Correlation Ranking with respect to target column '{target_col}' ({method}):\n\n"
            f"{ranking_df.to_markdown(index=False)}\n\n"
            f"```python\n{code}\n```"
        )
    except Exception as e:
        return f"Error ranking correlations: {str(e)}"