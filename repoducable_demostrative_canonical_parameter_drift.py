#!/usr/bin/env python3
"""
Parameter-Induced Drift — canonical reproducible demonstration

This script generates an irreversible contraction process with time-varying
coupling k(t), then fits mis-specified exponential models over increasing
observation windows to show systematic drift of the fitted decay rate λ(T).

It reproduces the structural bias described in Section IX of:
"A Dissipative Channel Formalism Across Optics, Open Quantum Systems,
 Thermodynamics, Information Theory and Ageing".

Minimal dependencies: numpy, scipy, matplotlib

Run:
    python parameter_induced_drift.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Model specification
# -----------------------------

def kappa(t, k0=0.2, alpha=0.6, omega=0.15):
    """
    Time-varying effective coupling κ(t).
    This breaks the semigroup / constant-rate assumption.
    """
    return k0 * (1.0 + alpha * np.sin(omega * t))


def cumulative_rate(t, k0=0.2, alpha=0.6, omega=0.15):
    """
    Integral ∫₀ᵗ κ(t') dt' computed analytically.
    """
    return k0 * (t - (alpha / omega) * (np.cos(omega * t) - 1.0))


def order_functional(t, O0=1.0, **kwargs):
    """
    True irreversible contraction law:
        O(t) = O0 * exp( - ∫₀ᵗ κ(t') dt' )
    """
    return O0 * np.exp(-cumulative_rate(t, **kwargs))


# -----------------------------
# Mis-specified exponential fit
# -----------------------------

def exponential_model(t, O0, lam):
    """Mis-specified constant-rate model O(t) = O0 * exp(-λ t)."""
    return O0 * np.exp(-lam * t)


def fit_effective_rate(t, O):
    """Fit a single exponential and return fitted λ."""
    popt, _ = curve_fit(exponential_model, t, O, p0=(O[0], 0.1), maxfev=10000)
    return popt[1]


# -----------------------------
# Demonstration
# -----------------------------

def main():
    # Time grid
    t_max = 100.0
    n_pts = 4000
    t = np.linspace(0.0, t_max, n_pts)

    # Generate true contraction
    O_true = order_functional(t)

    # Observation windows
    windows = np.linspace(10.0, t_max, 30)
    fitted_lambdas = []

    for T in windows:
        mask = t <= T
        lam_hat = fit_effective_rate(t[mask], O_true[mask])
        fitted_lambdas.append(lam_hat)

    fitted_lambdas = np.array(fitted_lambdas)

    # -------------------------
    # Plot results
    # -------------------------

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # (A) True contraction
    axs[0].plot(t, O_true, lw=2)
    axs[0].set_title("True contraction with time-varying κ(t)")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("O(t)")

    # (B) Drift of fitted decay rate
    axs[1].plot(windows, fitted_lambdas, "o-", lw=2)
    axs[1].set_title("Parameter-induced drift of fitted λ(T)")
    axs[1].set_xlabel("Observation window T")
    axs[1].set_ylabel("Fitted λ(T)")

    plt.tight_layout()
    plt.show()

    # Print diagnostic summary
    print("Parameter-induced drift demonstration")
    print("-----------------------------------")
    print(f"Initial fitted λ ≈ {fitted_lambdas[0]:.4f}")
    print(f"Final fitted λ   ≈ {fitted_lambdas[-1]:.4f}")
    print("Drift magnitude  ≈", fitted_lambdas[-1] - fitted_lambdas[0])


if __name__ == "__main__":
    main()
