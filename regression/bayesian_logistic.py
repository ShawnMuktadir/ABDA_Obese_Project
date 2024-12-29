import arviz as az
import matplotlib.pyplot as plt
import pymc as pm


def bayesian_logistic_regression(df, target, predictors):
    """
    Performs Bayesian Logistic Regression using PyMC.
    """
    X = df[predictors].values
    y = df[target].values

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