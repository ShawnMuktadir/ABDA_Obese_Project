import pandas as pd
from sklearn.model_selection import train_test_split

from utils.obesity.obesity_percentile_utils import process_obesity_with_75th_percentile
from utils.predictor_utils.age import process_age_column
from utils.predictor_utils.data import impute_data_value
from utils.predictor_utils.education import preprocess_education
from utils.predictor_utils.gender import impute_gender
from utils.predictor_utils.income import impute_income
from utils.predictor_utils.race import impute_race_ethnicity
from utils.validate_predictors import validate_predictors


def preprocess_data(df):
    """
    Full pipeline for preprocessing data.
    """
    print("[Start] Preprocessing pipeline.")
    df = process_age_column(df)
    df = impute_income(df)
    df = impute_data_value(df)
    df = impute_gender(df)
    df = impute_race_ethnicity(df)
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

def split_combined_data(combined_data, target, sample_size=1000, test_split=0.2):
    """
    Splits the combined data into training and testing sets after stratified sampling.
    Arguments:
        combined_data: The processed and combined dataset.
        target: Target column name (e.g., 'Obese').
        sample_size: Number of samples for stratified sampling.
        test_split: Proportion of the data to use for testing.
    Returns:
        train_data: Training dataset.
        test_data: Testing dataset.
        predictors: List of predictors to use for modeling.
    """
    # Define sample size and adjust dynamically
    total_size = len(combined_data)
    if sample_size >= total_size:
        raise ValueError(f"[Error] sample_size ({sample_size}) cannot exceed total dataset size ({total_size})!")

    test_size = 1 - sample_size / total_size
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

    # Define predictors
    behavioral_predictors = ['No_Physical_Activity', 'Low_Fruit_Consumption',
                             'Low_Veg_Consumption']
    demographic_predictors = ['Age(years)', 'Income', 'Gender', 'Education']
    race_predictors = [col for col in combined_data.columns if col.startswith('Race_')]
    predictors = behavioral_predictors + demographic_predictors + race_predictors

    # Convert categorical columns to numeric
    if 'Education' in combined_data.columns and combined_data['Education'].dtype.name == 'category':
        combined_data['Education'] = combined_data['Education'].astype(float)
        print("[Info] Converted 'Education' to numeric type.")

    # Validate predictors
    print("[Debug] Predictors before validation:")
    print(predictors)
    validate_predictors(combined_data, predictors)

    # Debug predictors
    print("[Debug] Predictors in use:")
    print(predictors)

    return train_data, test_data, predictors

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