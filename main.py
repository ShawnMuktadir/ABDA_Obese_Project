import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from regression.bayesian_hierarchical_logistic import bayesian_hierarchical_logistic_regression
from regression.bayesian_logistic import bayesian_logistic_regression
from utils.predictor_utils.age import process_age_column
from utils.predictor_utils.data import impute_data_value
from utils.predictor_utils.gender import impute_gender
from utils.group.group_column_utils import ensure_numeric_group_columns, compute_group_proportions
from utils.predictor_utils.income import impute_income
from utils.obesity.obesity_utils import process_obesity_with_75th_percentile
from utils.predictor_utils.race import impute_race_ethnicity
from utils.visualizer_utils import ObesityVisualizer

from sklearn.model_selection import KFold
import pymc as pm

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
    return df


def perform_k_fold_cv(data, target, predictors, n_splits=5):
    """
    Perform K-Fold Cross-Validation for Bayesian Logistic Regression.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"[Info] Processing Fold {fold + 1}/{n_splits}")

        train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]

        # Run Bayesian Logistic Regression
        trace, model = bayesian_logistic_regression(train_data, target, predictors)

        # Posterior Predictive Check for the Training Data
        perform_posterior_predictive_check(trace, model, train_data, target, f"Fold {fold + 1} - Training")

        # Posterior Predictive Check for the Testing Data (and save the ppc)
        print(f"[Info] Sampling Posterior Predictive for Fold {fold + 1} - Testing")
        with model:
            ppc = pm.sample_posterior_predictive(trace)

        # Append Results
        fold_results.append({"fold": fold + 1, "trace": trace, "model": model, "ppc": ppc})

    return fold_results

def extract_y_obs_samples(posterior_predictive):
    """
    Extracts the 'y_obs' samples from posterior_predictive dynamically.
    Handles xarray.DataArray and numpy.ndarray.
    Arguments:
        posterior_predictive: The posterior predictive data structure.
    Returns:
        samples (numpy.ndarray): Extracted y_obs samples.
    """
    if 'y_obs' in posterior_predictive:
        y_obs_data = posterior_predictive['y_obs']
        # Handle xarray.DataArray or numpy.ndarray
        if hasattr(y_obs_data, 'values'):  # xarray.DataArray
            samples = y_obs_data.values
            print("[Debug] Extracted 'y_obs' samples from xarray (shape):", samples.shape)
        elif isinstance(y_obs_data, np.ndarray):  # numpy.ndarray
            samples = y_obs_data
            print("[Debug] Extracted 'y_obs' samples from ndarray (shape):", samples.shape)
        else:
            raise TypeError(f"[Error] 'y_obs' is of unsupported type: {type(y_obs_data)}")
        return samples
    else:
        raise KeyError("[Error] The observed variable 'y_obs' is not accessible in 'posterior_predictive'.")


def perform_posterior_predictive_check(trace, model, data, target, model_name):
    """
    Perform Posterior Predictive Check for a Bayesian model.
    Arguments:
        trace: The trace obtained from model fitting (pm.sample()).
        model: The PyMC model corresponding to the trace.
        data: The dataset being used for validation.
        target: The name of the target column in the dataset (e.g., 'Obese').
        model_name: A string to label plots with the model name.
    """
    print(f"[Info] Performing Posterior Predictive Check for {model_name}...")

    # Reactivate the model context
    with model:
        # Sample posterior predictive distributions
        ppc = pm.sample_posterior_predictive(trace)

    # Debugging: Print keys in ppc
    print("[Debug] Keys in ppc:", list(ppc.keys()))

    # Extract 'y_obs' samples using helper function
    posterior_predictive = ppc['posterior_predictive']
    samples = extract_y_obs_samples(posterior_predictive)

    # Compare observed vs. predicted distributions
    observed = data[target]
    print("[Debug] Observed data (first 10 values):", observed.head(10).values)

    # Compute mean prediction across chains and draws
    predicted = np.mean(samples, axis=(0, 1))  # Average over chains and draws
    print("[Debug] Predicted data (first 10 values):", predicted[:10])

    # Plot observed vs. predicted distributions
    plt.figure(figsize=(10, 6))
    plt.hist(observed, bins=10, alpha=0.5, label="Observed Data", density=True)
    plt.hist(predicted, bins=10, alpha=0.5, label="Posterior Predictive", density=True)
    plt.title(f"Posterior Predictive Check: {model_name}", fontsize=14)
    plt.xlabel("Target Variable", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'dir/Nutrition_Physical_Activity_and_Obesity.csv'
    df = pd.read_csv(file_path)

    # Debug: Check initial columns
    print("[Debug] Initial dataset columns:")
    print(df.columns)

    df = preprocess_data(df)

    # Debug: Check columns after preprocessing
    print("[Debug] Columns after preprocessing:")
    print(df.columns)

    combined_data = process_obesity_with_75th_percentile(df)

    # Debug: Inspect combined data columns
    print("[Debug] Combined data columns:")
    print(combined_data.columns)

    # Ensure group columns are numeric
    group_cols = ['LocationDesc', 'Race/Ethnicity']
    combined_data = ensure_numeric_group_columns(combined_data, group_cols)

    # Debug: Check numeric conversions
    print("[Debug] Numeric Group Columns:")
    for col in group_cols:
        print(f"Unique values in '{col}':", combined_data[col].unique())

    # Reduce dataset size with stratified sampling
    sample_size = min(1000, len(combined_data) - 1)  # Ensure sample size does not exceed dataset size
    test_size = 1 - sample_size / len(combined_data)

    if test_size <= 0.0 or test_size >= 1.0:
        test_size = 0.2  # Default to a 20% test split

    combined_data, _ = train_test_split(
        combined_data,
        test_size=test_size,
        stratify=combined_data['Obese'],
        random_state=42
    )

    target = 'Obese'
    behavioral_predictors = ['No_Physical_Activity', 'Low_Fruit_Consumption', 'Low_Veg_Consumption']
    demographic_predictors = ['Age(years)', 'Income', 'Gender']
    race_predictors = [col for col in combined_data.columns if col.startswith('Race_')]
    predictors = behavioral_predictors + demographic_predictors + race_predictors

    # Debug predictors
    print("[Debug] Predictors in use:")
    print(predictors)

    # Visualize Obesity Distribution
    visualizer = ObesityVisualizer()
    visualizer.plot_obesity_distribution(combined_data)

    # Display Proportion of Obese Individuals
    proportion_obese = combined_data['Obese'].mean()
    print(f"Proportion of Obese individuals: {proportion_obese:.2%}")

    trace = bayesian_logistic_regression(combined_data, target, predictors)

    # Reload original values for mapping if needed
    original_df = pd.read_csv(file_path)
    combined_data['LocationDesc'] = original_df['LocationDesc']
    combined_data['Race/Ethnicity'] = original_df['Race/Ethnicity']

    # Convert to category type if not already
    if combined_data['LocationDesc'].dtype != 'category':
        combined_data['LocationDesc'] = combined_data['LocationDesc'].astype('category')
    if combined_data['Race/Ethnicity'].dtype != 'category':
        combined_data['Race/Ethnicity'] = combined_data['Race/Ethnicity'].astype('category')

    # Generate mappings
    location_mapping = {code: name for code, name in enumerate(combined_data['LocationDesc'].cat.categories)}
    race_mapping = {code: name for code, name in enumerate(combined_data['Race/Ethnicity'].cat.categories)}

    # Debug: Print the mappings
    print("[Debug] Location Mapping:")
    for code, name in location_mapping.items():
        print(f"{code}: {name}")

    print("[Debug] Race Mapping:")
    for code, name in race_mapping.items():
        print(f"{code}: {name}")

    # Compute proportions for LocationDesc
    location_proportions = compute_group_proportions(combined_data, 'LocationDesc', 'Obese')
    visualizer.plot_proportions_by_group(
        location_proportions,
        location_mapping,
        title="Proportion of Obese Individuals by Location",
        xlabel="Location",
        ylabel="Proportion Obese"
    )

    # Compute proportions for Race/Ethnicity
    race_categories = combined_data['Race/Ethnicity'].unique()  # Ensure unique categories are used
    race_mapping = {i: category for i, category in enumerate(race_categories)}

    # Debugging the mapping
    print("Race Mapping:", race_mapping)

    race_proportions = compute_group_proportions(combined_data, 'Race/Ethnicity', 'Obese')
    visualizer.plot_proportions_by_group(
        race_proportions,
        race_mapping,
        title="Proportion of Obese Individuals by Race/Ethnicity",
        xlabel="Race/Ethnicity",
        ylabel="Proportion Obese",
        figsize=(12, 6)  # Adjust size for race/ethnicity
    )

    # Perform Bayesian Hierarchical Logistic Regression
    print("[Info] Running Bayesian Hierarchical Logistic Regression...")
    trace_hierarchical = bayesian_hierarchical_logistic_regression(
        combined_data,
        target,
        predictors,
        group_cols
    )

    # Train Standard Model
    trace_standard, model_standard = bayesian_logistic_regression(combined_data, target, predictors)

    # Train Hierarchical Model
    trace_hierarchical, model_hierarchical = bayesian_hierarchical_logistic_regression(
        combined_data, target, predictors, group_cols
    )

    # Perform Posterior Predictive Check for Standard Model
    perform_posterior_predictive_check(trace_standard, model_standard,
                                       combined_data, target, "Standard Model")

    # Perform Posterior Predictive Check for Hierarchical Model
    perform_posterior_predictive_check(trace_hierarchical, model_hierarchical,
                                       combined_data, target, "Hierarchical Model")

    # Perform K-Fold Cross-Validation
    fold_results = perform_k_fold_cv(combined_data, target, predictors, n_splits=5)

    # Posterior Predictive Check for the last fold
    last_result = fold_results[-1]
    ppc = last_result["ppc"]
    # Visualize Posterior Predictive Check
    observed = combined_data[target]

    # Access posterior predictive
    posterior_predictive = ppc['posterior_predictive']

    # Extract 'y_obs' samples using helper function
    samples = extract_y_obs_samples(posterior_predictive)

    # Compute mean prediction across chains and draws
    predicted = np.mean(samples, axis=(0, 1))  # Average over chain and draw dimensions
    print("[Debug] Predicted data (first 10 values):", predicted[:10])

    # Compute mean prediction across chain and draw dimensions
    predicted = np.mean(samples, axis=(0, 1))  # Average over chains and draws

    plt.figure(figsize=(10, 6))
    plt.hist(observed, bins=10, alpha=0.5, label="Observed Data", density=True)
    plt.hist(predicted, bins=10, alpha=0.5, label="Posterior Predictive", density=True)
    plt.title("Posterior Predictive Check", fontsize=14)
    plt.xlabel("Target Variable", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
