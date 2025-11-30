import numpy as np
def bic_log_evidence(loglik, k, N):
    """
    Compute BIC and approximate log model evidence from a fitted model.

    Parameters
    ----------
    loglik : float
        Maximum log-likelihood log L_max for the model.
    k : int
        Number of free parameters in the model.
    N : int
        Number of scalar observations used to fit the model.

    Returns
    -------
    log_evidence : float
        Approximate log p(y | m) using the BIC (Laplace) approximation.
    bic : float
        Bayesian Information Criterion value.
    """
    bic = k * np.log(N) - 2.0 * loglik
    log_evidence = -0.5 * bic
    return log_evidence, bic
