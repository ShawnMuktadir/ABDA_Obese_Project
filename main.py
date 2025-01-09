import pymc as pm

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
from utils.df_utils import prepare_combined_data, reload_and_convert_columns, split_combined_data
from utils.location_race_map import generate_mappings
from utils.visualize_utils import VisualizerUtils


def main():
    file_path = 'dir/Nutrition_Physical_Activity_and_Obesity.csv'

    # Step 1: Preprocessing and integration
    combined_data, original_df = prepare_combined_data(file_path)

    # Reload original values for mapping if needed
    group_predictors_columns = ['LocationDesc', 'Race/Ethnicity', 'Education']
    combined_data = reload_and_convert_columns(combined_data, original_df, group_predictors_columns)

    # Step 2: Stratified sampling and train-test split
    target = 'Obese'
    train_data, test_data, predictors = split_combined_data(combined_data, target)

    # Visualize Obesity Distribution
    visualizer = VisualizerUtils()
    visualizer.plot_obesity_distribution(combined_data)

    # Generate mappings for LocationDesc, Race/Ethnicity, and Education
    location_mapping, race_mapping, education_mapping = generate_mappings(combined_data)

    # Visualize proportions for LocationDesc
    visualizer.compute_and_visualize_proportions(
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
    visualizer.compute_and_visualize_proportions(
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

    education_mapping = {
        -1.0: "Unknown",
        0.0: "Less than high school",
        1.0: "High school graduate",
        2.0: "Some college",
        3.0: "College graduate"
    }
    print("[Debug] Education Mapping:", education_mapping)

    # Visualize Obesity by Education Level
    visualizer.compute_and_visualize_proportions(
        combined_data,
        group_col='Education',
        target='Obese',
        mapping=education_mapping,
        visualizer=visualizer,
        title="Proportion of Obese Individuals by Education Level",
        xlabel="Education Level",
        ylabel="Proportion Obese",
        figsize=(12, 6)
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
        train_data, target, predictors, group_predictors_columns
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
                                                group_cols=group_predictors_columns)

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
