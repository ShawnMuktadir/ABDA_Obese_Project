import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymc as pm


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