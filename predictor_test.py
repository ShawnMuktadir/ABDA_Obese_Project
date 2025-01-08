import pandas as pd

from utils.df_utils import split_combined_data
from utils.predictor.age import process_age_column
from utils.predictor.education import preprocess_education
from utils.predictor.gender import impute_gender
from utils.predictor.income import impute_income
from utils.predictor.race import impute_race_ethnicity
from utils.validate_predictors import validate_predictors


def process_and_calculate_predictors(df):
    """
    Processes and calculates all predictors for the dataset.
    Arguments:
        df: Input DataFrame containing raw data.
    Returns:
        df: DataFrame with processed predictors.
    """
    print("[Info] Starting predictors processing...")

    # Process Age predictor
    if 'Age(years)' in df.columns:
        df = process_age_column(df)
        print("[Debug] Processed 'Age(years)' column values:")
        print(df['Age(years)'].head(10))  # Debug output for processed values
    else:
        print("[Warning] 'Age(years)' column not found. Adding placeholder values.")
        df['Age(years)'] = -1  # Placeholder
        print("[Debug] Added 'Age(years)' column with default value -1.")

    # Process Gender predictor
    if 'Gender' in df.columns:
        df = impute_gender(df)
        print("[Debug] Gender predictor processed.")

    # Process Income predictor
    if 'Income' in df.columns:
        df = impute_income(df)
        print("[Debug] Income predictor processed.")

    # Process Education predictor
    if 'Education' in df.columns:
        df = preprocess_education(df)
        print("[Debug] Education predictor processed.")

    # Process Race predictor
    if 'Race/Ethnicity' in df.columns:
        df = impute_race_ethnicity(df)
        print("[Debug] Race predictor processed.")

    print("[Info] All predictors processed successfully.")
    return df

def define_predictors(df):
    """
    Define predictors for modeling based on the processed data.
    Returns:
        predictors: List of all predictors (numeric + one-hot encoded).
        categorical_predictors: List of categorical predictors.
    """
    print("[Info] Defining predictors...")

    # Debug: Print available columns
    print("[Debug] DataFrame columns before defining predictors:", df.columns.tolist())

    # Behavioral predictors
    behavioral_predictors = ['No_Physical_Activity', 'Low_Fruit_Consumption', 'Low_Veg_Consumption']

    # Demographic predictors
    demographic_predictors = ['Age(years)', 'Income', 'Gender', 'Education']

    # Race predictors (one-hot encoded)
    race_predictors = [col for col in df.columns if col.startswith('Race_')]
    if race_predictors:
        print("[Info] Race predictors successfully identified:", race_predictors)
    else:
        print("[Warning] No race predictors found after processing.")

    # Categorical predictors
    categorical_predictors = [col for col in demographic_predictors if col in df.columns and pd.api.types.is_categorical_dtype(df[col])]

    # Combine all predictors
    predictors = behavioral_predictors + demographic_predictors + race_predictors

    # Debug: Final predictors
    print("[Debug] Numeric + One-Hot Encoded Predictors:", predictors)
    print("[Debug] Categorical Predictors:", categorical_predictors)

    return predictors, categorical_predictors

def validate_and_prepare_data(df, predictors, categorical_predictors):
    print("[Info] Validating predictors...")
    missing_predictors = [col for col in predictors if col not in df.columns]
    if missing_predictors:
        available_columns = df.columns.tolist()
        raise KeyError(f"[Error] Missing predictors in DataFrame: {missing_predictors}. "
                       f"Available columns: {available_columns}")
    # Validate numeric predictors
    numeric_predictors = [col for col in predictors if col not in categorical_predictors]
    validate_predictors(df, numeric_predictors)

    # Ensure all predictors are numeric or categorical as expected
    for predictor in numeric_predictors:
        if not pd.api.types.is_numeric_dtype(df[predictor]):
            raise TypeError(f"[Error] Predictor '{predictor}' is not numeric. Current type: {df[predictor].dtype}")

    for predictor in categorical_predictors:
        if not pd.api.types.is_categorical_dtype(df[predictor]):
            raise TypeError(
                f"[Error] Predictor '{predictor}' is not categorical. Current type: {df[predictor].dtype}")

        print("[Info] Predictors validated successfully.")

def summarize_predictor_statistics(data, predictors, categorical_predictors=None):
    """
    Summarize descriptive statistics for numeric, categorical, and one-hot encoded predictors.
    Arguments:
        data: DataFrame containing the dataset.
        predictors: List of predictor column names.
        categorical_predictors: Optional list of categorical predictor column names.
    """
    print("[Info] Summarizing predictor statistics...")

    # Debug: Ensure predictors are present
    print("[Debug] Available predictors in DataFrame:", predictors)
    print("[Debug] DataFrame columns:", data.columns.tolist())

    # Separate predictors into numeric, categorical, and one-hot encoded groups
    numeric_predictors = [
        col for col in predictors
        if pd.api.types.is_numeric_dtype(data[col]) and not col.startswith('Race_')
    ]
    if categorical_predictors is None:
        categorical_predictors = [
            col for col in predictors if pd.api.types.is_categorical_dtype(data[col])
        ]
    one_hot_encoded_predictors = [col for col in predictors if col.startswith('Race_')]

    print("[Debug] Identified numeric predictors:", numeric_predictors)
    print("[Debug] Identified categorical predictors:", categorical_predictors)
    print("[Debug] Identified one-hot encoded predictors:", one_hot_encoded_predictors)

    # Summarize numeric predictors
    numeric_stats = data[numeric_predictors].describe().T if numeric_predictors else None

    # Summarize categorical predictors
    categorical_summary = {}
    for col in categorical_predictors:
        print(f"[Debug] Processing categorical predictor: {col}")
        categorical_summary[col] = {
            "Unique": data[col].nunique(),
            "Top": data[col].mode()[0] if not data[col].mode().empty else "N/A",
            "Top Frequency": data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0
        }
    categorical_stats = (
        pd.DataFrame(categorical_summary).T if categorical_summary else None
    )

    if categorical_stats is None:
        print("[Warning] No categorical predictors found. Ensure categorical columns are processed correctly.")

    # Summarize one-hot encoded predictors
    one_hot_summary = {}
    for col in one_hot_encoded_predictors:
        print(f"[Debug] Processing one-hot encoded predictor: {col}")
        one_hot_summary[col] = {
            "Sum": data[col].sum(),
            "Mean": data[col].mean()
        }
    one_hot_stats = (
        pd.DataFrame(one_hot_summary).T if one_hot_summary else None
    )

    print("[Debug] Numeric predictor statistics summary:")
    print(numeric_stats)
    print("[Debug] Categorical predictor statistics summary:")
    print(categorical_stats)
    print("[Debug] One-hot encoded predictor statistics summary:")
    print(one_hot_stats)

    return numeric_stats, categorical_stats, one_hot_stats


def process_obesity_data(df):
    """
    Process the dataset to include Obese classification and integrate indirect questions.
    Arguments:
        df: The input DataFrame to process.
    Returns:
        combined_data: A DataFrame with Obese classification and relevant indirect questions integrated.
    """
    print("[Info] Processing obesity data with 75th percentile logic...")

    # Filter data for obesity-related questions
    obesity_df = df[df['Class'] == 'Obesity / Weight Status']
    obesity_data = obesity_df[obesity_df['Question'] == 'Percent of adults aged 18 years and older who have obesity']
    obesity_data['Data_Value'] = obesity_data['Data_Value'].fillna(obesity_data['Data_Value'].mean())
    quantile_threshold = obesity_data['Data_Value'].quantile(0.75)
    obesity_data['Obese'] = (obesity_data['Data_Value'] >= quantile_threshold).astype(int)

    # Define indirect questions
    indirect_questions = [
        'Percent of adults who engage in no leisure-time physical activity',
        'Percent of adults who report consuming fruit less than one time daily',
        'Percent of adults who report consuming vegetables less than one time daily'
    ]
    indirect_data = df[df['Question'].isin(indirect_questions)]

    # Merge obesity data with indirect questions
    combined_data = pd.merge(
        obesity_data[['YearStart', 'LocationDesc', 'Obese']],
        indirect_data[['YearStart', 'LocationDesc', 'Question', 'Data_Value']],
        on=['YearStart', 'LocationDesc'],
        how='left'
    )

    # Pivot table to transform indirect questions into columns
    combined_data = combined_data.pivot_table(
        index=['YearStart', 'LocationDesc', 'Obese'],
        columns='Question',
        values='Data_Value'
    ).reset_index()

    # Rename columns for clarity
    combined_data.rename(columns={
        'Percent of adults who engage in no leisure-time physical activity': 'No_Physical_Activity',
        'Percent of adults who report consuming fruit less than one time daily': 'Low_Fruit_Consumption',
        'Percent of adults who report consuming vegetables less than one time daily': 'Low_Veg_Consumption'
    }, inplace=True)

    # Add demographic and race predictors
    demographic_cols = ['Age(years)', 'Income', 'Gender', 'Education']
    race_cols = [col for col in df.columns if col.startswith('Race_')]

    all_predictors = demographic_cols + race_cols
    combined_data = pd.merge(
        combined_data,
        df[['YearStart', 'LocationDesc'] + all_predictors].drop_duplicates(),
        on=['YearStart', 'LocationDesc'],
        how='left'
    )

    # Debugging the final structure
    print("[Info] Indirect questions processed and included in combined data.")
    print("[Debug] Combined data columns after merging demographic and race predictors:", combined_data.columns.tolist())

    return combined_data

def main():
    file_path = 'dir/Nutrition_Physical_Activity_and_Obesity.csv'

    # Step 1: Load and preprocess data
    df = pd.read_csv(file_path)
    df = process_and_calculate_predictors(df)

    # Step 2: Process Obesity Data
    combined_data = process_obesity_data(df)

    # Step 3: Split Data and Define Predictors
    train_data, test_data, predictors, categorical_predictors = split_combined_data(
        combined_data, target='Obese'
    )

    # Step 4: Validate and Prepare Data
    validate_and_prepare_data(combined_data, predictors, categorical_predictors)

    # Step 5: Summarize Predictors
    summarize_predictor_statistics(combined_data, predictors, categorical_predictors)

    print("[Info] Workflow completed successfully.")

if __name__ == "__main__":
    main()

