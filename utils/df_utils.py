import pandas as pd
from sklearn.model_selection import train_test_split

from utils.obesity.obesity_percentile_utils import process_obesity_with_75th_percentile
from utils.predictor.age import process_age_column
from utils.predictor.data import impute_data_value
from utils.predictor.education import preprocess_education
from utils.predictor.gender import preprocess_gender
from utils.predictor.income import preprocess_income
from utils.predictor.race import preprocess_race_ethnicity
from utils.validate_predictors import validate_predictors


def preprocess_data(df):
    """
    Full pipeline for preprocessing data.
    """
    print("[Start] Preprocessing pipeline.")
    df = process_age_column(df)
    df = preprocess_income(df)
    df = impute_data_value(df)
    df = preprocess_gender(df)
    df = preprocess_race_ethnicity(df)
    df = preprocess_education(df)
    return df

def prepare_combined_data(path):
    """
    Prepares the combined data by processing the dataset, adding demographics,
    and encoding categorical columns.
    Arguments:
        path: Path to the dataset CSV file.
    Returns:
        combined_data: Processed and combined dataset.
        original_df: Original DataFrame (for mapping purposes).
    """
    # Load dataset
    file_path = path
    df = pd.read_csv(file_path)

    # Preprocessing
    df = preprocess_data(df)

    # Process obesity and integrate indirect questions
    combined_data = process_obesity_with_75th_percentile(df)

    # Ensure numeric encoding for group columns
    group_cols = ['LocationDesc', 'Race/Ethnicity', 'Education']
    for col in group_cols:
        if col in combined_data.select_dtypes(include=['category']).columns:
            combined_data[col] = combined_data[col].cat.codes
        elif col in combined_data.columns:
            combined_data[col] = combined_data[col].astype('category').cat.codes

    print("[Debug] Unique values in 'LocationDesc':", combined_data['LocationDesc'].unique())
    print("[Debug] Unique values in 'Race/Ethnicity':", combined_data['Race/Ethnicity'].unique())
    print("[Debug] Unique values in 'Education':", combined_data['Education'].unique())

    return combined_data, df

def split_combined_data(combined_data, target, predictors, categorical_predictors, sample_size=None, test_split=0.2):
    """
    Splits the combined data into training and testing sets after stratified sampling.
    Arguments:
        combined_data: The processed and combined dataset.
        target: Target column name (e.g., 'Obese').
        predictors: List of predictors to use for modeling.
        categorical_predictors: List of categorical predictors.
        sample_size: Number of samples for stratified sampling.
        test_split: Proportion of the data to use for testing.
    Returns:
        train_data: Training dataset.
        test_data: Testing dataset.
        predictors: List of predictors to use for modeling.
    """
    # Default sample_size to the full dataset size if not provided
    if sample_size is None:
        sample_size = len(combined_data)

    # Check if the sample size is valid
    total_size = len(combined_data)
    if sample_size > total_size:
        raise ValueError(f"[Error] sample_size ({sample_size}) cannot exceed total dataset size ({total_size})!")

    # Calculate test size
    test_size = 1 - sample_size / total_size
    if test_size <= 0.0 or test_size >= 1.0:
        print(f"[Warning] Adjusting invalid test_size ({test_size}) to default value ({test_split}).")
        test_size = test_split

    print(f"[Info] Stratified Sampling: Using sample_size={sample_size}, test_size={test_size:.2f}")

    # Perform stratified sampling
    combined_data, _ = train_test_split(
        combined_data,
        test_size=test_size,
        stratify=combined_data[target],
        random_state=42
    )

    # Split into training and testing datasets
    train_data, test_data = train_test_split(
        combined_data,
        test_size=test_split,
        stratify=combined_data[target],
        random_state=42
    )

    # Debug: Check predictors
    print("[Debug] Predictors defined within split_combined_data:", predictors)
    print("[Debug] Categorical predictors defined within split_combined_data:", categorical_predictors)

    return train_data, test_data

def reload_and_convert_columns(combined_data, original_df, columns):
    """
    Reloads original values for specified columns and converts them to categorical type.
    Arguments:
        combined_data: DataFrame to update.
        original_df: Original DataFrame to reload values from.
        columns: List of column names to reload and convert.
    Returns:
        Updated combined_data with reloaded and converted columns.
    """
    for column in columns:
        if column in original_df.columns:
            combined_data[column] = original_df[column]  # Reload original values

            # Convert to category type if not already
            if combined_data[column].dtype != 'category':
                combined_data[column] = combined_data[column].astype('category')
                print(f"[Info] Converted '{column}' to categorical type.")
        else:
            print(f"[Warning] Column '{column}' not found in original data!")
    return combined_data