import numpy as np

def simulate_dcm(A, C, T, u_amp, std_process_noise, std_observation_noise, random_state=42):
    """
    Simulate a Dynamic Causal Model (DCM) time series.

    Parameters
    ----------
    A : Array of shape (n_regions, n_regions) - effective connectivity matrix.
    C : Array of shape (n_regions, 1) - input driver matrix.
    T : int
        Number of time steps to simulate.
    amp : float
        Amplitude scaling factor for the stimulus when it is applied.
    std_process_noise : float
        Standard deviation of the process noise.
    std_observation_noise : float
        Standard deviation of the observation noise.
    random_state : int, optional
        Seed for the random number generator.
    Returns
    -------
    x : Array of shape (T, n_regions)
        Simulated hidden states over time.
    y : Array of shape (T, n_regions)
        Simulated observations over time.
    u : Array of shape (T, 1)
        Input stimulus over time.
    
    """
    
    rng= np.random.default_rng(random_state)
    n_regions = A.shape[0] # Number of brain regions
    u = np.zeros((T, 1))
    u[T // 4:T // 2] = u_amp
    u[3 * T // 4:] = u_amp

    x = np.zeros((T, n_regions))
    y = np.zeros((T, n_regions))

    for t in range(T-1): #state evolution
        w_t = rng.normal(0, std_process_noise, size=n_regions)
        x[t + 1] = A @ x[t] + (C @ u[t]).ravel() + w_t # discrete time state equation

    for t in range(T): #observation loop
        v_t = rng.normal(0, std_observation_noise, size=n_regions)
        y[t] = x[t] + v_t # observation equation

    return x, y, u