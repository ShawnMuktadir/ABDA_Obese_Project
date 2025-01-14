from sklearn.preprocessing import StandardScaler

from utils.predictor.location_desc import preprocess_location

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymc as pm

# Load the dataset
file_path = 'dir/Nutrition_Physical_Activity_and_Obesity.csv'
df = pd.read_csv(file_path)

# Import preprocessing functions
from utils.predictor.age import process_age_column
from utils.predictor.data import impute_data_value
from utils.predictor.education import preprocess_education
from utils.predictor.gender import preprocess_gender
from utils.predictor.income import preprocess_income
from utils.predictor.race import preprocess_race_ethnicity

# Process individual columns for missing data
df = process_age_column(df)
df = impute_data_value(df)
df = preprocess_education(df)
df = preprocess_gender(df)
df = preprocess_income(df)
df = preprocess_race_ethnicity(df)
df = preprocess_location(df)

# Map relevant questions to concise column names
question_to_column = {
    'Percent of adults who engage in no leisure-time physical activity': 'No_Physical_Activity',
    'Percent of adults aged 18 years and older who have an overweight classification': 'Overweight_Rate',
    'Percent of adults who achieve at least 300 minutes a week of moderate-intensity aerobic physical activity or 150 minutes a week of vigorous-intensity aerobic physical activity (or an equivalent combination)': 'High_Physical_Activity',
    'Percent of adults who achieve at least 150 minutes a week of moderate-intensity aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic physical activity and engage in muscle-strengthening activities on 2 or more days a week': 'Physical_Activity_With_Strength',
    'Percent of adults who achieve at least 150 minutes a week of moderate-intensity aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic physical activity (or an equivalent combination)': 'Moderate_Physical_Activity',
    'Percent of adults who engage in muscle-strengthening activities on 2 or more days a week': 'Strength_Training',
    'Percent of adults who report consuming fruit less than one time daily': 'Low_Fruit_Consumption',
    'Percent of adults who report consuming vegetables less than one time daily': 'Low_Veg_Consumption'
}

# Include mapped questions in the dataset
mapped_questions = df[df['Question'].isin(question_to_column.keys())]
for question, column_name in question_to_column.items():
    mapped_questions.loc[mapped_questions['Question'] == question, 'Question'] = column_name

# Check if 'YearEnd' exists in the dataset
pivot_index_columns = ['LocationDesc', 'YearStart']
mapped_pivot = mapped_questions.pivot_table(index=pivot_index_columns, columns='Question', values='Data_Value').reset_index()

# Debug: Show the pivot table structure
print("[Debug] Pivot table created with index columns:", pivot_index_columns)

# Include demographic and mapped question columns in the final dataset
important_columns = list(mapped_pivot.columns) + ['Age(years)', 'Income']
categorical_columns = ['Gender', 'Education', 'LocationDesc']
pivot_df = df.copy()

for col in important_columns:
    if col not in pivot_df.columns:
        pivot_df[col] = mapped_pivot[col] if col in mapped_pivot.columns else np.nan

# Debug: Check categorical columns
for col in categorical_columns:
    if col in df.columns:
        pivot_df[col] = df[col]
        print(f"[Debug] {col} column preview before processing:")
        print(pivot_df[col].value_counts(dropna=False))
    else:
        print(f"[Warning] {col} column is missing in the DataFrame.")

# Ensure numeric columns are processed separately
numeric_columns = [col for col in pivot_df.columns if col not in categorical_columns and not col.startswith('Race_')]

# Convert numeric columns to numeric and handle errors
pivot_df[numeric_columns] = pivot_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill missing values with column mean for numeric columns
pivot_df[numeric_columns] = pivot_df[numeric_columns].fillna(pivot_df[numeric_columns].mean())

# Ensure categorical columns have no NaN values
for col in categorical_columns:
    if col in pivot_df.columns:
        if pd.api.types.is_categorical_dtype(pivot_df[col]):
            if 'Unknown' not in pivot_df[col].cat.categories:
                pivot_df[col] = pivot_df[col].cat.add_categories('Unknown')
            pivot_df[col] = pivot_df[col].fillna('Unknown')
        else:
            pivot_df[col] = pivot_df[col].fillna('Unknown')

# Calculate the correlation matrix only for numeric columns
correlation_matrix = pivot_df[numeric_columns].corr()

# Save the correlation matrix for supervisor review
correlation_matrix_path = 'dir/correlation_matrix_with_all_columns_and_mapped_questions.csv'
correlation_matrix.to_csv(correlation_matrix_path)
print(f"Complete correlation matrix saved to {correlation_matrix_path}")

# Define thresholds
high_corr_threshold = 0.2  # Lower threshold to capture weaker correlations
redundancy_threshold = 0.9  # Relax redundancy threshold to retain more features

# Ensure mapped questions and demographics are included
feature_corr = correlation_matrix.mean(axis=1).sort_values(ascending=False)
high_corr_features = feature_corr[feature_corr.abs() > high_corr_threshold].index.tolist()

# Identify and remove redundant features
redundant_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > redundancy_threshold:
            redundant_features.add(correlation_matrix.columns[j])

# Debug: Output redundant features
print("[Debug] Features removed due to high correlation:", redundant_features)

# Combine all features: high_corr_features, race_columns, and important_columns
race_columns = [col for col in pivot_df.columns if col.startswith('Race_')]
final_features = list(set(high_corr_features + race_columns + important_columns + categorical_columns))

# Remove Race/Ethnicity as it is redundant due to one-hot encoding
if 'Race/Ethnicity' in final_features:
    final_features.remove('Race/Ethnicity')

# Ensure no redundant features
final_features = [feature for feature in final_features if feature not in redundant_features]

# Output results
print("\nFinal selected features after ensuring inclusion of one-hot encoded Race/Ethnicity columns and other features:")
print(final_features)

# Ensure categorical columns are in final_features
for col in ['Gender', 'Education', 'LocationDesc']:
    if col not in final_features and col in pivot_df.columns:
        print(f"[Debug] Adding '{col}' to final features.")
        final_features.append(col)

# Save the filtered dataset with updated selected features
selected_df = pivot_df[final_features]
output_path = 'dir/filtered_nutrition_data_with_race_and_mapped_questions.csv'
selected_df.to_csv(output_path, index=False)
print(f"Filtered dataset saved to {output_path}")

# Show aggregated histogram for Race columns
if race_columns:
    selected_df[race_columns].sum().plot(kind='bar', title='Histogram of Race/Ethnicity (One-Hot Encoded)', xlabel='Race Categories', ylabel='Frequency', rot=45)
    plt.grid(True)
    plt.show()

# Plot histograms for Gender and Education
for col in categorical_columns:
    if col in selected_df.columns:
        selected_df[col].value_counts().plot(kind='bar', title=f'Histogram of {col}', xlabel=col, ylabel='Frequency', rot=45, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Plot histograms for other selected features
for feature in final_features:
    if feature in selected_df.columns and feature not in race_columns and feature not in categorical_columns:
        plt.figure()
        selected_df[feature].plot(kind='hist', bins=30, title=f'Histogram of {feature}', fontsize=8)
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def bayesian_logistic_regression(df, target, predictors):
    """
    Performs Bayesian Logistic Regression using PyMC.
    """
    X = df[predictors].values
    y = df[target].values

    # Debug: Show predictor names and corresponding data
    print("[Debug] Predictor Mapping and Sample Data:")
    for idx, predictor in enumerate(predictors):
        print(f"Column {idx}: {predictor}")

    # Debug: Show the first few rows of the predictor matrix
    print("[Debug] X sample data (first 5 rows):")
    print(pd.DataFrame(X, columns=predictors).head())

    # Debug: Inspect data types and content
    print("[Debug] X shape:", X.shape)
    print("[Debug] X data types:", pd.DataFrame(X).dtypes)
    print("[Debug] X sample data:\n", pd.DataFrame(X).head())
    print("[Debug] y shape:", y.shape)

    # Ensure all columns in X are numeric
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(
            f"[Error] Predictors matrix 'X' contains non-numeric data. Column data types:\n{pd.DataFrame(X).dtypes}")

    # Check for NaNs in X (for pandas DataFrame)
    if pd.isna(X).any().any():
        print("[Debug] NaN values detected in X. Replacing with column means.")
        X = X.fillna(X.mean())

    # Convert to NumPy array for further processing
    X = X.values

    # Debug: Check predictors after handling NaNs
    print("[Debug] X after replacing NaNs (first 5 rows):")
    print(X[:5])

    # Check for NaNs or infinities in X and y
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("[Error] NaN values detected in predictors or target.")
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError("[Error] Infinity values detected in predictors or target.")

    # Build the PyMC model
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        logits = intercept + pm.math.dot(X, beta)
        y_obs = pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        # Sampling
        trace = pm.sample(draws=1000, tune=2000, chains=2, cores=2, target_accept=0.95)

    # Debug: Posterior Summary
    summary = az.summary(trace, hdi_prob=0.95)
    print("[Debug] Posterior summary:")
    print(summary)

    # Plotting
    az.plot_forest(trace, var_names=["beta"], combined=True)
    plt.title("Posterior Coefficients")
    plt.show()

    return trace, model

def clean_predictors(df, predictors):
    """
    Preprocesses predictors to ensure all are numeric.
    - Checks for missing values or invalid types.
    """
    # Ensure only predictors are kept
    predictors_cleaned = [col for col in df.columns if col in predictors]

    # Select processed predictors
    X = df[predictors_cleaned]

    # Ensure all columns in X are numeric
    if not np.issubdtype(X.values.dtype, np.number):
        raise TypeError("[Error] Predictors matrix 'X' contains non-numeric data after encoding.")

    # Check for NaN values
    if X.isnull().any().any():
        raise ValueError("[Error] NaN values detected in predictors after preprocessing.")

    return X

# Ensure categorical columns are converted to one-hot encoding
def preprocess_categorical(df, categorical_columns):
    """
    Converts categorical columns to one-hot encoded format.
    """
    # Use one-hot encoding for categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Convert all boolean columns to numeric (0/1)
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded

def preprocess_data(df, predictors):
    """
    Preprocesses the DataFrame to ensure all predictors are numeric.
    Converts categorical variables to numeric using one-hot encoding or label encoding.
    """
    processed_df = df.copy()
    for col in predictors:
        if not pd.api.types.is_numeric_dtype(processed_df[col]):
            print(f"[Info] Converting non-numeric column '{col}' to numeric values.")
            if processed_df[col].nunique() <= 2:  # Binary categorical column
                # Label encode binary categorical column (e.g., 'Male', 'Female' -> 0, 1)
                processed_df[col] = processed_df[col].astype('category').cat.codes
            else:
                # One-hot encode for multi-category
                one_hot = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, one_hot], axis=1)
                processed_df.drop(columns=[col], inplace=True)
    return processed_df

def bayesian_logistic_regression(df, target, predictors):
    """
    Performs Bayesian Logistic Regression using PyMC.
    """

    # Preprocess predictors and target column
    df = preprocess_data(df, predictors)

    # Align predictors with available columns in df
    aligned_predictors = [col for col in predictors if col in df.columns]

    # Debug: Check for non-numeric values in predictors
    for col in aligned_predictors:
        invalid_rows = df[~df[col].apply(lambda x: pd.api.types.is_number(x))]
        if not invalid_rows.empty:
            print(f"[Debug] Invalid values detected in column '{col}':")
            print(invalid_rows[[col]].head())  # Print a few invalid values for debugging
            raise ValueError(f"[Error] Column '{col}' contains non-numeric values. Please clean the data.")

    # Preprocess predictors: Convert to numeric (ensures errors are handled)
    X = df[aligned_predictors].apply(pd.to_numeric, errors='coerce')

    # Convert DataFrame to NumPy array (ensure float type)
    X = X.astype(float).to_numpy()

    # Check and handle NaN values
    if np.isnan(X).any():
        print("[Debug] NaN values detected in X. Replacing with column means.")
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])

    # Scale predictors
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Validate target
    y = df[target].values
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("[Error] Target variable 'y' contains non-binary or invalid values.")

    # Debug: Check predictors and target
    print("[Debug] X shape:", X.shape, "y shape:", y.shape)

    # Build PyMC model
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=0.5, shape=X.shape[1])  # Stronger prior
        intercept = pm.Normal("intercept", mu=0, sigma=0.5)  # Stronger prior
        logits = intercept + pm.math.dot(X, beta)
        y_obs = pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        # Debug: Check model log-probability
        logp_func = model.compile_logp()
        logp_value = logp_func(model.initial_point())
        print(f"[Debug] Model log-probability: {logp_value}")

        # Sampling
        trace = pm.sample(draws=1000, tune=2000, chains=2, cores=2, target_accept=0.95)

    # Debug: Posterior summary
    summary = az.summary(trace, hdi_prob=0.95)
    print("[Debug] Posterior summary:")
    print(summary)

    # Plot posterior distributions
    az.plot_forest(trace, var_names=["beta"], combined=True)
    plt.title("Posterior Coefficients")
    plt.show()

    return trace, model

if __name__ == '__main__':
    # Define target and predictors for Bayesian Logistic Regression
    target_column = 'Obesity_Binary'

    # Define Obesity_Binary based on Overweight_Rate
    if 'Overweight_Rate' in selected_df.columns:
        selected_df['Obesity_Binary'] = (selected_df['Overweight_Rate'] > 30).astype(int)  # Example threshold
        print("[Info] 'Obesity_Binary' column created based on 'Overweight_Rate' > 30.")
    else:
        raise KeyError("[Error] 'Overweight_Rate' column not found in selected_df. Cannot create 'Obesity_Binary'.")

    # Debug: Confirm 'Obesity_Binary' creation
    print("[Debug] 'Obesity_Binary' column value counts:")
    print(selected_df['Obesity_Binary'].value_counts())

    predictors = [col for col in final_features if col != target_column]

    # Perform Bayesian Logistic Regression
    trace, model = bayesian_logistic_regression(df=selected_df, target=target_column, predictors=predictors)