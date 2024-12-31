import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def bayesian_hierarchical_logistic_regression(df, target, predictors, group_cols):
    """
    Performs Bayesian Logistic Regression with Hierarchical Modeling using PyMC.
    Converts group column names into data directly within the function.
    """
    # Ensure consistent group encoding across all folds or datasets
    group_indices = {}
    for group in group_cols:
        df[group] = pd.Categorical(df[group]).codes  # Convert to categorical codes
        group_indices[group] = pd.Categorical(df[group]).codes

    # Extract predictors and target
    X = df[predictors].values
    y = df[target].values

    # Debug: Show predictor names and corresponding data
    print("[Debug] Predictor Mapping and Sample Data:")
    for idx, predictor in enumerate(predictors):
        print(f"Column {idx}: {predictor}")

    # Debug: Show the first few rows of the predictor matrix
    print("[Debug] X sample data (first 5 rows):")
    print(pd.DataFrame(X, columns=predictors).head())

    # Validate predictors and target are numeric
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(
            f"[Error] Predictors matrix 'X' contains non-numeric data. Data types:\n{df[predictors].dtypes}")
    if not np.issubdtype(y.dtype, np.number):
        raise TypeError(f"[Error] Target vector 'y' is not numeric. Data type: {y.dtype}")

    with pm.Model() as hierarchical_model:
        # Global priors
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        intercept = pm.Normal("intercept", mu=0, sigma=1)

        # Group-specific priors
        group_effects = {}
        for group, indices in group_indices.items():
            n_categories = len(np.unique(indices))  # Unique categories for the group
            group_effects[group] = pm.Normal(f"{group}_effect", mu=0, sigma=1, shape=n_categories)
            print(f"[Debug] Group '{group}' has {n_categories} unique categories.")

        # Hierarchical logits
        logits = intercept + pm.math.dot(X, beta)
        for group, indices in group_indices.items():
            logits += group_effects[group][indices]

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        # Sampling
        trace = pm.sample(draws=1000, tune=2000, chains=2, cores=2, target_accept=0.95, return_inferencedata=True)

    # Debug: Posterior Summary
    summary = az.summary(trace, hdi_prob=0.95)
    print("[Debug] Posterior summary with hierarchical effects:")
    print(summary)

    # Visualize posterior distributions of group effects
    if "posterior" in trace:
        group_effects_vars = [var for var in trace["posterior"].data_vars if "_effect" in var]
        for group_effect in group_effects_vars:
            az.plot_posterior(trace, var_names=[group_effect])
            plt.title(f"Posterior Distribution for {group_effect}")
            plt.show()

        # Forest plot of group effects
        az.plot_forest(trace, var_names=group_effects_vars, combined=True)
        plt.title("Forest Plot of Group Effects")
        plt.show()
    else:
        print("[Warning] Posterior data not found in the trace object.")

    return trace, hierarchical_model