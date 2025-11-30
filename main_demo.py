import numpy as np

from mini_dcm.simulate import simulate_dcm
from mini_dcm.models import (
    effective_connectivity,
    model_mask_chain_true,
    model_mask_missing_2to3,
    model_mask_wrong_3to1,
)
from mini_dcm.fit import fit_model
from mini_dcm.evidence import bic_log_evidence


def softmax(log_evidences):
    """
    Convert log-evidences into normalized posterior probabilities.
    """
    le = np.array(log_evidences, dtype=float)
    le = le - np.max(le)
    w = np.exp(le)
    return w / w.sum()


def main():
    #Ground-truth connectivity
    A_true, C_true = effective_connectivity()

    #Simulate data from the true model
    T = 200
    u_amp = 1.0
    std_process = 0.05
    std_obs = 0.05

    x, y, u = simulate_dcm(
        A_true, C_true,
        T=T,
        u_amp=u_amp,
        std_process_noise=std_process,
        std_observation_noise=std_obs,
        random_state=0,
    )

    #Define candidate model structure
    models = {
        "M1_chain_true":      model_mask_chain_true(),
        "M2_missing_2to3":    model_mask_missing_2to3(),
        "M3_wrong_3to1":      model_mask_wrong_3to1(),
    }

    names = []
    log_evidences = []

    #Fit each model and compute evidence
    for name, mask in models.items():
        A_hat, loglik, k, N = fit_model(y, u, mask)
        log_ev, bic = bic_log_evidence(loglik, k, N)

        print(f"Model: {name}")
        print("A_hat:")
        print(A_hat)
        print(f"log-likelihood = {loglik:.2f}")
        print(f"log-evidence  ≈ {log_ev:.2f}")
        print(f"BIC           = {bic:.2f}")
        print("-" * 40)

        names.append(name)
        log_evidences.append(log_ev)

    #Posterior model probabilities
    post = softmax(log_evidences)

    print("Posterior model probabilities (equal priors):")
    for name, p in zip(names, post):
        print(f"{name:20s}: {p:.3f}")


if __name__ == "__main__":
    main()
