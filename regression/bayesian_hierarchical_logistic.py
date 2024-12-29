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
    # Extract data for group columns
    group_indices = {col: pd.Categorical(df[col]).codes for col in group_cols}

    # Extract predictors and target
    X = df[predictors].values
    y = df[target].values

    with pm.Model() as hierarchical_model:
        # Global priors
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        intercept = pm.Normal("intercept", mu=0, sigma=1)

        # Group-specific priors
        group_effects = {}
        for group, indices in group_indices.items():
            group_effects[group] = pm.Normal(f"{group}_effect", mu=0, sigma=1, shape=len(np.unique(indices)))

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
    group_effects_vars = []
    for var in trace.posterior.data_vars:
        if "_effect" in var:
            group_effects_vars.append(var)
    for group_effect in group_effects_vars:
        az.plot_posterior(trace, var_names=[group_effect])
        plt.title(f"Posterior Distribution for {group_effect}")
        plt.show()

    # Forest plot of group effects
    az.plot_forest(trace, var_names=group_effects_vars, combined=True)
    plt.title("Forest Plot of Group Effects")
    plt.show()

    return trace, hierarchical_model