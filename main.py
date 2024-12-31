import numpy as np
import pandas as pd
import pymc as pm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from regression.bayesian_hierarchical_logistic import bayesian_hierarchical_logistic_regression
from regression.bayesian_logistic import bayesian_logistic_regression
from utils.group.group_column_utils import ensure_numeric_group_columns, compute_group_proportions
from utils.obesity.obesity_utils import process_obesity_with_75th_percentile
from utils.predictor_utils.age import process_age_column
from utils.predictor_utils.data import impute_data_value
from utils.predictor_utils.gender import impute_gender
from utils.predictor_utils.income import impute_income
from utils.predictor_utils.race import impute_race_ethnicity
from utils.visualizer_utils import ObesityVisualizer

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


def perform_k_fold_cv(data, target, predictors, n_splits=5,
                      model_type="standard", group_cols=None):
    """
    Perform K-Fold Cross-Validation for Bayesian models.
    Arguments:
        data: The dataset to split into folds.
        target: The target variable for the model (e.g., 'Obese').
        predictors: The list of predictors to use in the model.
        n_splits: Number of folds for K-Fold Cross-Validation.
        model_type: Either 'standard' or 'hierarchical'.
        group_cols: Group columns for hierarchical models (if applicable).
    Returns:
        fold_results: List of dictionaries containing trace, model, and metrics for each fold.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []  # This list will store results for all folds

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"[Info] Processing Fold {fold + 1}/{n_splits}...")
        train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]

        # Train the appropriate model
        if model_type == "standard":
            trace, model = bayesian_logistic_regression(train_data, target, predictors)
        elif model_type == "hierarchical":
            trace, model = bayesian_hierarchical_logistic_regression(train_data, target, predictors, group_cols)
        else:
            raise ValueError(f"[Error] Unsupported model type: {model_type}")

        # Store the model (Important for posterior predictive checks)
        # Line: Save model and trace for each fold
        fold_results.append({
            "fold": fold + 1,
            "trace": trace,
            "model": model,  # Store the model for reactivation later
            "test_data": test_data,  # Save fold-specific test data for evaluation
        })

        # Compute accuracy (Optional for immediate feedback)
        with model:
            ppc = pm.sample_posterior_predictive(trace, random_seed=42)
            # Debugging: Print keys
            print("[Debug] Keys in ppc:", list(ppc.keys()))
            if 'posterior_predictive' in ppc and 'y_obs' in ppc['posterior_predictive']:
                y_obs_samples = ppc['posterior_predictive']['y_obs']
            elif 'y_obs' in ppc:
                y_obs_samples = ppc['y_obs']
            else:
                raise KeyError("[Error] Could not find 'y_obs' in posterior predictive samples.")

            # Debug: Print shape
            print(f"[Debug] y_obs_samples shape: {y_obs_samples.shape}, Test data shape: {test_data.shape}")
            # Ensure predicted matches observed
            num_test_samples = len(test_data)
            predicted = np.mean(y_obs_samples[..., :num_test_samples], axis=(0, 1))
            observed = test_data[target].values
            if predicted.shape[0] != observed.shape[0]:
                raise ValueError(f"[Error] Shape mismatch: Predicted shape {predicted.shape}, Observed shape {observed.shape}")

            # Compute accuracy
            accuracy = float(np.mean((predicted >= 0.5) == observed))

            # Debug: Print calculated accuracy
            print(f"[Debug] Fold {fold + 1} Accuracy: {accuracy:.2%}")

            # Add accuracy to fold results
            fold_results[-1]["accuracy"] = accuracy
            print(f"[Info] Fold {fold + 1}/{n_splits} - Accuracy: {accuracy:.2%}")

    # Return all fold results, including trace and model for each fold
    return fold_results  # Line: Return trace and model

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

def perform_posterior_predictive_check(trace, model, data, target, model_name, is_test=False):
    """
    Perform Posterior Predictive Check for a Bayesian model (Training/Testing).
    Arguments:
        trace: The trace obtained from model fitting (pm.sample()).
        model: The PyMC model corresponding to the trace.
        data: The dataset being used for validation (training or testing).
        target: The name of the target column in the dataset (e.g., 'Obese').
        model_name: A string to label plots with the model name.
        is_test: A boolean flag to indicate whether this is a testing PPC.
    """
    print(f"[Info] Performing Posterior Predictive Check for {'Testing' if is_test else 'Training'} Data - {model_name}...")

    # Reactivate the model context
    with model:
        # Sample posterior predictive distributions
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])

    # Debugging: Print keys in ppc
    print("[Debug] Keys in ppc:", list(ppc.keys()))

    # Extract 'y_obs' samples using helper function
    posterior_predictive = ppc['posterior_predictive']
    samples = extract_y_obs_samples(posterior_predictive)

    # Compare observed vs. predicted distributions
    observed = data[target]
    print(f"[Debug] Observed {'Testing' if is_test else 'Training'} data (first 10 values):", observed.head(10).values)

    # Compute mean prediction across chains and draws
    predicted = np.mean(samples, axis=(0, 1))  # Average over chains and draws
    print(f"[Debug] Predicted {'Testing' if is_test else 'Training'} data (first 10 values):", predicted[:10])

    # Plot observed vs. predicted distributions
    plt.figure(figsize=(10, 6))
    plt.hist(observed, bins=10, alpha=0.5, label=f"Observed {'Testing' if is_test else 'Training'} Data", density=True)
    plt.hist(predicted, bins=10, alpha=0.5, label=f"Posterior Predictive for {'Testing' if is_test else 'Training'}", density=True)
    plt.title(f"Posterior Predictive Check: {model_name} ({'Testing' if is_test else 'Training'} Data)", fontsize=14)
    plt.xlabel("Target Variable", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def display_k_fold_results(standard_results, hierarchical_results):
    """
    Display K-Fold CV results in tabular format.
    Arguments:
        standard_results: Results from Standard Model.
        hierarchical_results: Results from Hierarchical Model.
    """
    # Extract accuracy values for each model
    standard_accuracies = [result["accuracy"] for result in standard_results]
    hierarchical_accuracies = [result["accuracy"] for result in hierarchical_results]

    # Create a DataFrame for fold-wise results
    results_df = pd.DataFrame({
        "Fold": range(1, len(standard_accuracies) + 1),
        "Standard Model Accuracy": standard_accuracies,
        "Hierarchical Model Accuracy": hierarchical_accuracies,
    })

    # Display the results
    print("[Info] Fold-Wise Results:")
    print(results_df)

    return results_df

def visualize_k_fold_results(standard_results, hierarchical_results):
    """
    Visualize K-Fold CV results for Standard and Hierarchical Models.
    Arguments:
        standard_results: Results from Standard Model.
        hierarchical_results: Results from Hierarchical Model.
    """
    # Extract accuracy values
    standard_accuracies = [result["accuracy"] for result in standard_results]
    hierarchical_accuracies = [result["accuracy"] for result in hierarchical_results]
    folds = range(1, len(standard_accuracies) + 1)

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(folds, standard_accuracies, width=0.4, label="Standard Model", align="center", alpha=0.7)
    plt.bar([x + 0.4 for x in folds], hierarchical_accuracies, width=0.4, label="Hierarchical Model", align="center", alpha=0.7)
    plt.title("K-Fold Cross-Validation Results", fontsize=16)
    plt.xlabel("Fold", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(folds)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def visualize_overall_posterior_predictive(ppc, observed, model_name):
    """
    Visualize overall posterior predictive distribution compared to observed data.
    Arguments:
        ppc: Posterior predictive samples.
        observed: Observed target variable (actual data).
        model_name: Name of the model for labeling the plot.
    """
    if 'posterior_predictive' in ppc and 'y_obs' in ppc['posterior_predictive']:
        samples = ppc['posterior_predictive']['y_obs']
    elif 'y_obs' in ppc:
        samples = ppc['y_obs']
    else:
        raise KeyError("[Error] Could not find 'y_obs' in posterior predictive samples.")

    # Aggregate predictions
    predicted = np.mean(samples, axis=(0, 1))  # Average over chains and draws

    # Plot observed vs. predicted
    plt.figure(figsize=(10, 6))
    plt.hist(observed, bins=10, alpha=0.5, label="Observed Data", density=True)
    plt.hist(predicted, bins=10, alpha=0.5, label="Posterior Predictive", density=True)
    plt.title(f"Posterior Predictive Check: {model_name}", fontsize=14)
    plt.xlabel("Target Variable", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_fold_posterior_predictive(ppc, observed, fold, model_name):
    """
    Visualize posterior predictive distribution for a specific fold.
    Arguments:
        ppc: Posterior predictive samples.
        observed: Observed target variable for the test dataset.
        fold: Fold number for labeling the plot.
        model_name: Name of the model for labeling the plot.
    """
    if 'posterior_predictive' in ppc and 'y_obs' in ppc['posterior_predictive']:
        samples = ppc['posterior_predictive']['y_obs']
    elif 'y_obs' in ppc:
        samples = ppc['y_obs']
    else:
        raise KeyError("[Error] Could not find 'y_obs' in posterior predictive samples.")

    # Aggregate predictions for the fold
    predicted = np.mean(samples, axis=(0, 1))  # Average over chains and draws

    # Plot observed vs. predicted for the fold
    plt.figure(figsize=(10, 6))
    plt.hist(observed, bins=10, alpha=0.5, label="Observed Data", density=True)
    plt.hist(predicted, bins=10, alpha=0.5, label="Posterior Predictive", density=True)
    plt.title(f"Posterior Predictive Check: Fold {fold} ({model_name})", fontsize=14)
    plt.xlabel("Target Variable", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'dir/Nutrition_Physical_Activity_and_Obesity.csv'
    df = pd.read_csv(file_path)

    # Target column
    target = 'Obese'  # Define the target column name explicitly

    # Preprocessing pipeline
    df = preprocess_data(df)
    combined_data = process_obesity_with_75th_percentile(df)

    # Ensure group columns are numeric
    group_cols = ['LocationDesc', 'Race/Ethnicity']
    combined_data = ensure_numeric_group_columns(combined_data, group_cols)

    # Define sample size and adjust test size dynamically
    sample_size = 1000  # Set the sample size for stratified sampling
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

    # Split into training and testing data
    train_data, test_data = train_test_split(
        combined_data,
        test_size=0.2,  # 20% of the sampled data for testing
        stratify=combined_data[target],
        random_state=42
    )

    # Define predictors
    behavioral_predictors = ['No_Physical_Activity', 'Low_Fruit_Consumption',
                             'Low_Veg_Consumption']
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
    location_proportions = compute_group_proportions(combined_data,
                                                     'LocationDesc', 'Obese')
    visualizer.plot_proportions_by_group(
        location_proportions,
        location_mapping,
        title="Proportion of Obese Individuals by Location",
        xlabel="Location",
        ylabel="Proportion Obese"
    )

    # Compute proportions for Race/Ethnicity
    race_categories = combined_data['Race/Ethnicity'].unique()
    # Ensure unique categories are used
    race_mapping = {i: category for i, category in enumerate(race_categories)}

    # Debugging the mapping
    print("Race Mapping:", race_mapping)

    race_proportions = compute_group_proportions(combined_data,
                                                 'Race/Ethnicity', 'Obese')
    visualizer.plot_proportions_by_group(
        race_proportions,
        race_mapping,
        title="Proportion of Obese Individuals by Race/Ethnicity",
        xlabel="Race/Ethnicity",
        ylabel="Proportion Obese",
        figsize=(12, 6)  # Adjust size for race/ethnicity
    )

    # Train Standard Model on Training Data
    trace_standard, model_standard = bayesian_logistic_regression(train_data, target,
                                                                  predictors)

    # Perform Posterior Predictive Check for Standard Model (Training)
    perform_posterior_predictive_check(trace_standard, model_standard, train_data,
                                       target, "Standard Model")

    # Perform Posterior Predictive Check for Standard Model (Testing)
    perform_posterior_predictive_check(trace_standard, model_standard, test_data,
                                       target, "Standard Model", is_test=True)

    # Perform Overall Posterior Predictive Check for Standard Model
    print("[Info] Performing Overall Posterior Predictive Check for Standard Model...")
    with model_standard:
        ppc_standard = pm.sample_posterior_predictive(trace_standard, random_seed=42)
    visualize_overall_posterior_predictive(ppc_standard, combined_data[target],
                                           "Standard Model")

    # Train Hierarchical Model on Training Data
    trace_hierarchical, model_hierarchical = bayesian_hierarchical_logistic_regression(
        train_data, target, predictors, group_cols
    )

    # Perform Posterior Predictive Check for Hierarchical Model (Training)
    perform_posterior_predictive_check(trace_hierarchical, model_hierarchical,
                                       train_data, target, "Hierarchical Model")

    # Perform Posterior Predictive Check for Hierarchical Model (Testing)
    perform_posterior_predictive_check(trace_hierarchical, model_hierarchical,
                                       test_data, target, "Hierarchical Model", is_test=True)

    # Perform Overall Posterior Predictive Check for Hierarchical Model
    print("[Info] Performing Overall Posterior Predictive Check for Hierarchical Model...")
    with model_hierarchical:
        ppc_hierarchical = pm.sample_posterior_predictive(trace_hierarchical, random_seed=42)
    visualize_overall_posterior_predictive(ppc_hierarchical, combined_data[target],
                                           "Hierarchical Model")

    # Perform K-Fold Cross-Validation for Standard Model
    print("[Info] Performing K-Fold Cross-Validation for Standard Model...")
    standard_cv_results = perform_k_fold_cv(train_data, target, predictors,
                                            n_splits=5, model_type="standard")

    # Visualize Fold-wise Posterior Predictive Checks for Standard Model
    print("[Info] Visualizing Fold-wise Posterior Predictive Checks for Standard Model...")
    for fold_result in standard_cv_results:
        with fold_result["model"]:  # Activate model context
            fold_ppc = pm.sample_posterior_predictive(fold_result["trace"], random_seed=42)
        visualize_fold_posterior_predictive(
            fold_ppc,
            fold_result["test_data"][target],  # Use fold-specific test data
            fold_result["fold"],
            "Standard Model"
        )

    # Perform K-Fold Cross-Validation for Hierarchical Model
    print("[Info] Performing K-Fold Cross-Validation for Hierarchical Model...")
    hierarchical_cv_results = perform_k_fold_cv(train_data, target, predictors, n_splits=5,
                                                model_type="hierarchical",
                                                group_cols=group_cols)

    # Visualize Fold-wise Posterior Predictive Checks for Hierarchical Model
    print("[Info] Visualizing Fold-wise Posterior Predictive Checks for Hierarchical Model...")
    for fold_result in hierarchical_cv_results:
        with fold_result["model"]:  # Activate model context
            fold_ppc = pm.sample_posterior_predictive(fold_result["trace"], random_seed=42)
        visualize_fold_posterior_predictive(
            fold_ppc,
            fold_result["test_data"][target],  # Use fold-specific test data
            fold_result["fold"],
            "Hierarchical Model"
        )

    # Display Tabular Results
    k_fold_results_df = display_k_fold_results(standard_cv_results, hierarchical_cv_results)

    print("[Info] K-Fold Cross-Validation Results:")
    print(k_fold_results_df)
    # k_fold_results_df.to_csv("k_fold_results.csv", index=False)
    # print("[Info] K-Fold results saved to 'k_fold_results.csv'.")

    # Visualize Results
    visualize_k_fold_results(standard_cv_results, hierarchical_cv_results)

    print("[Info] Workflow completed successfully!")

if __name__ == "__main__":
    main()
