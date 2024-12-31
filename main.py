import pandas as pd
import pymc as pm
from sklearn.model_selection import train_test_split

from regression.bayesian_hierarchical_logistic import bayesian_hierarchical_logistic_regression
from regression.bayesian_logistic import bayesian_logistic_regression
from utils.bayesian_model_utils import (
    perform_k_fold_cv,
    perform_posterior_predictive_check,
    display_k_fold_results,
    visualize_k_fold_results,
    visualize_overall_posterior_predictive,
    visualize_fold_posterior_predictive
)
from utils.group.group_column_utils import (
    ensure_numeric_group_columns
)
from utils.location_race_map import generate_mappings, compute_and_visualize_proportions
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
    location_mapping, race_mapping = generate_mappings(combined_data)
    # Visualize proportions for LocationDesc
    compute_and_visualize_proportions(
        combined_data,
        group_col='LocationDesc',
        target='Obese',
        mapping=location_mapping,
        visualizer=visualizer,
        title="Proportion of Obese Individuals by Location",
        xlabel="Location",
        ylabel="Proportion Obese"
    )

    # Visualize proportions for Race/Ethnicity
    compute_and_visualize_proportions(
        combined_data,
        group_col='Race/Ethnicity',
        target='Obese',
        mapping=race_mapping,
        visualizer=visualizer,
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
