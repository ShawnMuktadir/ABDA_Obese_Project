import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

from regression.bayesian_hierarchical_logistic import bayesian_hierarchical_logistic_regression
from regression.bayesian_logistic import bayesian_logistic_regression

# Utility Function: Perform K-Fold Cross-Validation
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

# Utility Function: Extract 'y_obs' Samples
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

# Utility Function: Perform Posterior Predictive Check
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

# Utility Function: Display K-Fold Results
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

# Utility Function: Visualize K-Fold Results
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

# Utility Function: Visualize Overall Posterior Predictive
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

# Utility Function: Visualize Fold Posterior Predictive
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