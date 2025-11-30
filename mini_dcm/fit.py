import numpy as np

def fit_model(y, u, mask):
    T, n_regions = y.shape
    x_previous = y[:-1]  # x_t
    x_next = y[1:]       # x_{t+1}
    u_previous = u[:-1]  # u_t
    A_hat = np.zeros((n_regions, n_regions))
    total_rss = 0.0
    total_observations = 0
    total_params = 0

    for i in range(n_regions):
        parents = np.where(mask[i])[0]
        xi=np.hstack([x_previous[:, parents], u_previous])
        yi=x_next[:, i]
        beta, *_ = np.linalg.lstsq(xi, yi, rcond=None)
        k_i = len(beta) - 1
        A_hat[i, parents] = beta[:k_i]
        residuals = yi - xi @ beta
        total_rss += np.sum(residuals**2)
        total_observations += len(yi)
        total_params += len(beta)

    sigma2 = total_rss / total_observations
    sigma2 = max(sigma2, 1e-8) #to avoid log(0)
    log_likelihood = -0.5 * total_observations * (np.log(2 * np.pi * sigma2) + 1)
    return A_hat, log_likelihood, total_params, total_observations