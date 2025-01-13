import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.df_utils import split_combined_data
from utils.predictor.age import process_age_column
from utils.predictor.education import preprocess_education
from utils.predictor.gender import preprocess_gender
from utils.predictor.income import preprocess_income
from utils.predictor.location_desc import preprocess_location
from utils.predictor.race import preprocess_race_ethnicity

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
        df = preprocess_gender(df)
        print("[Debug] Gender predictor processed.")

    # Process Income predictor
    if 'Income' in df.columns:
        df = preprocess_income(df)
        print("[Debug] Income predictor processed.")

    # Process Education predictor
    if 'Education' in df.columns:
        df = preprocess_education(df)
        print("[Debug] Education predictor processed.")

    # Process Race predictor
    if 'Race/Ethnicity' in df.columns:
        df = preprocess_race_ethnicity(df)
        print("[Debug] Race predictor processed.")

    # Process LocationDesc predictor
    if 'LocationDesc' in df.columns:
        df = preprocess_location(df)
        print("[Debug] LocationDesc predictor processed.")

    # Process YearStart predictor
    if 'YearStart' in df.columns:
        year_min = df['YearStart'].min()
        df['Years Since Min'] = df['YearStart'] - year_min
        print(f"[Debug] Created 'Years Since Min' column with min year: {year_min}")
        print(df[['YearStart', 'Years Since Min']].head(5))
    else:
        print("[Warning] 'YearStart' column not found in the DataFrame.")

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

    # Behavioral predictors
    behavioral_predictors = ['No_Physical_Activity', 'Low_Fruit_Consumption', 'Low_Veg_Consumption']

    # Demographic predictors
    demographic_numeric = ['Age(years)', 'Income']
    demographic_categorical = ['Gender', 'Education', 'LocationDesc']

    # Adding derived features to predictors
    derived_features = ['Sedentary_to_Physical_Ratio', 'Income_to_Calorie_Ratio']

    # Ensure 'Gender', 'Education', and 'LocationDesc' are processed as categorical
    categorical_predictors = [
        col for col in demographic_categorical
        if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype)
    ]

    # Validate numeric demographic predictors
    numeric_predictors = [
        col for col in demographic_numeric if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Combine numeric and categorical predictors
    demographic_predictors = numeric_predictors + categorical_predictors

    # Race predictors (one-hot encoded)
    race_predictors = [col for col in df.columns if col.startswith('Race_')]

    # Location and Year predictors
    location_predictors = ['LocationDesc'] if 'LocationDesc' in categorical_predictors else []
    year_predictors = ['Years Since Min'] if 'Years Since Min' in df.columns else []

    # Combine all predictors
    predictors = (behavioral_predictors + demographic_predictors + race_predictors
                  + location_predictors + year_predictors)

    # Debug: Final predictors
    print("[Debug] Final predictors list:", predictors)
    print("[Debug] Final predictors list length:", {len(predictors)})
    print("[Debug] Numeric + One-Hot Encoded Predictors:", numeric_predictors + race_predictors + year_predictors)
    print("[Debug] Categorical Predictors:", categorical_predictors)

    return predictors, categorical_predictors

def add_derived_features(df):
    """
    Adds derived features to the dataset:
    - Ratio of sedentary time to physical activity time.
    - Income-to-calorie-consumption ratio.
    """
    print("[Info] Adding derived features...")

    # Ratio of sedentary time to physical activity time
    if 'No_Physical_Activity' in df.columns and 'Low_Fruit_Consumption' in df.columns:
        # Using 'No_Physical_Activity' as sedentary time proxy and 'Low_Fruit_Consumption' for physical activity proxy
        df['Sedentary_to_Physical_Ratio'] = df['No_Physical_Activity'] / (
            df['Low_Fruit_Consumption'] + 1e-5)  # Avoid division by zero
        print("[Debug] Added 'Sedentary_to_Physical_Ratio' derived feature.")
    else:
        print("[Warning] Required columns for 'Sedentary_to_Physical_Ratio' not found.")

    # Income-to-calorie-consumption ratio
    if 'Income' in df.columns and 'Low_Fruit_Consumption' in df.columns and 'Low_Veg_Consumption' in df.columns:
        # Using 'Low_Fruit_Consumption' + 'Low_Veg_Consumption' as calorie consumption proxy
        df['Income_to_Calorie_Ratio'] = df['Income'] / (
            df['Low_Fruit_Consumption'] + df['Low_Veg_Consumption'] + 1e-5)  # Avoid division by zero
        print("[Debug] Added 'Income_to_Calorie_Ratio' derived feature.")
    else:
        print("[Warning] Required columns for 'Income_to_Calorie_Ratio' not found.")

    # Debug: Show first 5 rows of derived features
    if 'Sedentary_to_Physical_Ratio' in df.columns:
        print("[Debug] Sample values for 'Sedentary_to_Physical_Ratio':")
        print(df['Sedentary_to_Physical_Ratio'].head())
    if 'Income_to_Calorie_Ratio' in df.columns:
        print("[Debug] Sample values for 'Income_to_Calorie_Ratio':")
        print(df['Income_to_Calorie_Ratio'].head())

    return df


def validate_and_prepare_data(df, predictors, categorical_predictors):
    print("[Info] Validating predictors...")
    missing_predictors = [col for col in predictors if col not in df.columns]
    if missing_predictors:
        available_columns = df.columns.tolist()
        raise KeyError(f"[Error] Missing predictors in DataFrame: {missing_predictors}. "
                       f"Available columns: {available_columns}")

    # Separate numeric predictors correctly
    numeric_predictors = [
        col for col in predictors
        if col not in categorical_predictors and not col.startswith('Race_')
    ]
    print("[Debug] validate_and_prepare_data Numeric predictors for validation:", numeric_predictors)
    print("[Debug] validate_and_prepare_data Categorical predictors for validation:", categorical_predictors)

    # Validate numeric predictors
    for predictor in numeric_predictors:
        if not pd.api.types.is_numeric_dtype(df[predictor]):
            raise TypeError(f"[Error] Predictor '{predictor}' is not numeric. Current type: {df[predictor].dtype}")

    # Validate categorical predictors
    for predictor in categorical_predictors:
        if not isinstance(df[predictor].dtype, pd.CategoricalDtype):
            raise TypeError(
                f"[Error] Predictor '{predictor}' is not categorical. Current type: {df[predictor].dtype}")

    print("[Info] Predictors validated successfully.")

def summarize_predictor_statistics(data, predictors):
    """
    Summarize descriptive statistics for numeric, categorical, and one-hot encoded predictors.
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
    categorical_predictors = [
        col for col in predictors
        if isinstance(data[col].dtype, pd.CategoricalDtype)
    ]

    # Ensure 'Gender' and 'LocationDesc' are included in categorical predictors
    #additional_categoricals = ['Gender', 'LocationDesc']
    #for col in additional_categoricals:
    #    if col in data.columns and col not in categorical_predictors:
    #        if isinstance(data[col].dtype, pd.CategoricalDtype):
    #            categorical_predictors.append(col)

    one_hot_encoded_predictors = [col for col in predictors if col.startswith('Race_')]

    print("[Debug] summarize_predictor_statistics Identified numeric predictors:", numeric_predictors)
    print("[Debug] summarize_predictor_statistics Identified categorical predictors:", categorical_predictors)
    print("[Debug] summarize_predictor_statistics Identified one-hot encoded predictors:", one_hot_encoded_predictors)

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
        #print(f"[Debug] Processing one-hot encoded predictor: {col}")
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

    combined_data = combined_data.pivot_table(
        index=['YearStart', 'LocationDesc', 'Obese'],
        columns='Question',
        values='Data_Value'
    ).reset_index()

    # Ensure 'Years Since Min' is carried forward
    if 'Years Since Min' in df.columns:
        combined_data = pd.merge(
            combined_data,
            df[['YearStart', 'Years Since Min']].drop_duplicates(),
            on='YearStart',
            how='left'
        )

    # Rename columns for clarity
    combined_data.rename(columns={
        'Percent of adults who engage in no leisure-time physical activity': 'No_Physical_Activity',
        'Percent of adults who report consuming fruit less than one time daily': 'Low_Fruit_Consumption',
        'Percent of adults who report consuming vegetables less than one time daily': 'Low_Veg_Consumption'
    }, inplace=True)

    # Debug: Print columns before merging demographic data
    print("[Debug] Combined data columns before merging demographic and race predictors:", combined_data.columns.tolist())

    # Ensure unique 'LocationDesc' before merging
    df = df.drop_duplicates(subset=['YearStart', 'LocationDesc'])

    # Add demographic and race predictors
    demographic_cols = ['Age(years)', 'Income', 'Gender', 'Education']
    race_cols = [col for col in df.columns if col.startswith('Race_')]

    all_predictors = demographic_cols + race_cols

    # Drop duplicate 'LocationDesc' column from combined_data if it exists
    if combined_data.columns.duplicated().any():
        print("[Warning] Duplicated columns found. Removing duplicates...")
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

    combined_data = pd.merge(
        combined_data,
        df[['YearStart', 'LocationDesc'] + all_predictors],
        on=['YearStart', 'LocationDesc'],
        how='left'
    )

    # Debug: Check for duplicate columns after merging
    duplicate_columns = combined_data.columns[combined_data.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"[Warning] Duplicate columns found after merging: {duplicate_columns}")

    # Debugging the final structure
    print("[Info] Indirect questions processed and included in combined data.")
    print("[Debug] Combined data columns after merging demographic and race predictors:", combined_data.columns.tolist())
    print("[Debug] Combined data head:", combined_data.head(10))

    # Debug: Check for YearStart
    if 'YearStart' not in combined_data.columns:
        print("[Error] 'YearStart' column missing in combined_data after processing.")
    else:
        print("[Debug] 'YearStart' column exists in combined_data.")

    print("[Debug] Combined data columns after processing:")
    print(combined_data.columns.tolist())
    return combined_data

def analyze_unused_columns(df, used_columns):
    """
    Analyze unused columns and their potential as predictors.
    """
    print("[Info] Analyzing unused columns...")
    unused_columns = [col for col in df.columns if col not in used_columns]

    print("[Debug] Unused columns in DataFrame:", unused_columns)

    numeric_unused = df[unused_columns].select_dtypes(include='number').describe().T
    categorical_unused = df[unused_columns].select_dtypes(include=['object', 'category'])

    for col in categorical_unused.columns:
        print(f"[Debug] Unique values in {col}: {categorical_unused[col].unique()}")
        print(f"[Debug] Value counts for {col}:")
        print(categorical_unused[col].value_counts())

    return numeric_unused, categorical_unused

def analyze_dataset_completeness(data, numeric_predictors, categorical_predictors):
    """
    Analyze the total dataset size and the size of calculable data for modeling.
    Arguments:
        data: DataFrame containing the dataset.
        numeric_predictors: List of numeric predictor column names.
        categorical_predictors: List of categorical predictor column names.
    """
    print(f"[Info] Total dataset size: {data.shape[0]} rows, {data.shape[1]} columns")

    # Numeric completeness
    numeric_complete = data[numeric_predictors].notnull().all(axis=1)

    # Categorical completeness
    categorical_complete = ~data[categorical_predictors].isin(["Unknown"]).any(axis=1)

    # Combined completeness
    calculable_data_mask = numeric_complete & categorical_complete
    calculable_data_size = calculable_data_mask.sum()

    # Percentage of calculable data
    calculable_percentage = (calculable_data_size / data.shape[0]) * 100

    print(f"[Info] Total calculable data size: {calculable_data_size} rows")
    print(f"[Info] Percentage of calculable data: {calculable_percentage:.2f}%")

    return calculable_data_size, calculable_percentage

def visualize_predictor_completeness(data, predictors):
    """
    Visualize completeness of predictors as a bar chart.
    """
    print("[Info] Visualizing predictor completeness...")

    completeness = {
        predictor: data[predictor].notnull().mean() * 100 for predictor in predictors
    }

    # Convert completeness to DataFrame for visualization
    completeness_df = pd.DataFrame(list(completeness.items()), columns=['Predictor', 'Completeness (%)'])

    # Sort by completeness
    completeness_df.sort_values(by='Completeness (%)', ascending=False, inplace=True)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Completeness (%)', y='Predictor', data=completeness_df, palette='coolwarm')
    plt.title("Completeness of Predictors", fontsize=16)
    plt.xlabel("Completeness (%)", fontsize=12)
    plt.ylabel("Predictors", fontsize=12)
    plt.tight_layout()
    plt.show()

def visualize_numeric_predictor_relationships(data, numeric_predictors, target):
    """
    Create scatterplots for numeric predictors against the target variable.
    """
    print("[Info] Visualizing relationships for numeric predictors...")

    for predictor in numeric_predictors:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=predictor, y=target, data=data, alpha=0.6, color="blue")
        plt.title(f"Relationship of {predictor} with {target}", fontsize=16)
        plt.xlabel(predictor, fontsize=12)
        plt.ylabel(target, fontsize=12)
        plt.tight_layout()
        plt.show()

def visualize_categorical_predictor_relationships(data, categorical_predictors, target):
    """
    Create bar charts for categorical predictors against the target variable.
    """
    print("[Info] Visualizing relationships for categorical predictors...")

    for predictor in categorical_predictors:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=predictor, hue=target, data=data, palette='viridis')
        plt.title(f"Relationship of {predictor} with {target}", fontsize=16)
        plt.xlabel(predictor, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title=target)
        plt.tight_layout()
        plt.show()

def summarize_and_visualize_dataset(data, predictors, numeric_predictors, categorical_predictors, target):
    """
    Summarize and visualize dataset completeness and relationships with the target.
    """
    print("[Info] Summarizing and visualizing dataset...")

    # Visualize completeness
    visualize_predictor_completeness(data, predictors)

    # Visualize numeric predictors vs. target
    visualize_numeric_predictor_relationships(data, numeric_predictors, target)

    # Visualize categorical predictors vs. target
    visualize_categorical_predictor_relationships(data, categorical_predictors, target)

def summarize_data_matrix(df):
    """
    Summarize the data matrix structure and statistics.
    """
    print("[Summary] Data Matrix Overview:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\n[Columns Overview]")
    for col in df.columns:
        dtype = df[col].dtype
        unique_values = df[col].nunique()
        print(f"- {col}: Type={dtype}, Unique Values={unique_values}")
    print("\n[Target Variable]")
    #print(f"Target column ('Obese') distribution:\n{df['Obese'].value_counts(normalize=True)}")


def main():
    """
    Main function to execute the workflow.
    """
    file_path = 'dir/Nutrition_Physical_Activity_and_Obesity.csv'

    # Step 1: Load and preprocess data
    df = pd.read_csv(file_path)
    # Get the column names
    column_names = df.columns.tolist()

    # Print the column names
    print("Column names:")
    print(f"Number of columns: {len(column_names)}")
    for col in column_names:
        print(col)
    df = process_and_calculate_predictors(df)
    df = add_derived_features(df)

    summarize_data_matrix(df)

    # Step 2: Process Obesity Data
    combined_data = process_obesity_data(df)

    # Step 3: Define Predictors
    predictors, categorical_predictors = define_predictors(combined_data)

    # Step 4: Split Data and Validate Predictors
    train_data, test_data = split_combined_data(
        combined_data, target='Obese', predictors=predictors, categorical_predictors=categorical_predictors
    )

    # Analyze dataset completeness
    calculable_data_size, calculable_percentage = analyze_dataset_completeness(
        combined_data, predictors, categorical_predictors
    )

    # Output the analysis
    print("[Summary] Dataset Completeness Analysis:")
    print(f"  - Total rows: {combined_data.shape[0]}")
    print(f"  - Total columns: {combined_data.shape[1]}")
    print(f"  - Calculable rows: {calculable_data_size}")
    print(f"  - Percentage of calculable data: {calculable_percentage:.2f}%")

    print("[Debug] Calling validate_and_prepare_data with:")
    print("  - Numeric predictors:", [col for col in predictors if col not in categorical_predictors])
    print("  - Categorical predictors:", categorical_predictors)

    validate_and_prepare_data(combined_data, predictors, categorical_predictors)

    # Analyze unused columns
    numeric_unused, categorical_unused = analyze_unused_columns(combined_data, predictors)

    # Step 5: Summarize Predictors
    summarize_predictor_statistics(combined_data, predictors)

    # Separate numeric predictors for visualization
    numeric_predictors = [
        col for col in predictors if pd.api.types.is_numeric_dtype(combined_data[col])
    ]

    summarize_and_visualize_dataset(combined_data, predictors, numeric_predictors, categorical_predictors,
                                    target='Obese')

    print("[Info] Workflow completed successfully.")

if __name__ == "__main__":
    main()

