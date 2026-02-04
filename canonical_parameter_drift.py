#!/usr/bin/env python3
"""
canonical_parameter_drift.py

Minimal reproducible demonstration of parameter-induced drift.

We generate an irreversible process with a non-exponential hazard
(time-varying coupling), then fit a mis-specified exponential model
over different observation windows. The fitted decay rate drifts
systematically as a function of window length.

This illustrates how parametric rate models absorb structural mismatch
as drifting parameters.

Author: Gypsy Hors De Combat
License: CC0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Ground-truth irreversible dynamics
# -----------------------------

def true_coupling(t):
    """
    Time-dependent effective coupling κ(t).
    Increases slowly with time (senescence / memory / structured bath).
    """
    return 0.15 + 0.08 * np.log(1 + t)

def true_order(t):
    """
    Order functional O(t) = exp(-∫ κ(t) dt)
    Constructed numerically to avoid assuming exponential decay.
    """
    dt = t[1] - t[0]
    integral = np.cumsum(true_coupling(t)) * dt
    return np.exp(-integral)

# Time grid
t = np.linspace(0, 20, 2000)
O_true = true_order(t)

# -----------------------------
# Mis-specified exponential model
# -----------------------------

def exp_model(t, lam):
    return np.exp(-lam * t)

# -----------------------------
# Fit over sliding observation windows
# -----------------------------

window_lengths = np.linspace(2.0, 20.0, 25)   # increasing observation depth
fitted_rates = []

for Tmax in window_lengths:
    mask = t <= Tmax
    t_fit = t[mask]
    O_fit = O_true[mask]

    # Fit exponential decay O ≈ exp(-λ t)
    popt, _ = curve_fit(exp_model, t_fit, O_fit, p0=[0.2])
    fitted_rates.append(popt[0])

fitted_rates = np.array(fitted_rates)

# -----------------------------
# Plot results
# -----------------------------

plt.figure(figsize=(12, 4))

# Panel 1 — true dynamics vs exponential fits
plt.subplot(1, 2, 1)
plt.plot(t, O_true, 'k', lw=2, label='True irreversible dynamics')

for Tmax in [4, 8, 12, 16, 20]:
    mask = t <= Tmax
    popt, _ = curve_fit(exp_model, t[mask], O_true[mask], p0=[0.2])
    plt.plot(t, exp_model(t, popt[0]), '--', lw=1,
             label=f'Exp fit (T={Tmax:.0f}, λ={popt[0]:.3f})')

plt.xlabel('t')
plt.ylabel('O(t)')
plt.title('Mis-specified exponential fits')
plt.legend(fontsize=8)

# Panel 2 — parameter-induced drift
plt.subplot(1, 2, 2)
plt.plot(window_lengths, fitted_rates, 'o-', lw=2)
plt.xlabel('Observation window T')
plt.ylabel('Fitted decay rate λ̂(T)')
plt.title('Parameter-induced drift')

plt.tight_layout()
plt.show()

# -----------------------------
# Print diagnostic
# -----------------------------

print("\nParameter-induced drift demonstration\n")
for T, lam in zip(window_lengths[::4], fitted_rates[::4]):
    print(f"Window T = {T:5.2f}  ->  fitted λ = {lam:.4f}")

print("\nDrift range:")
print(f"min λ̂ = {fitted_rates.min():.4f}")
print(f"max λ̂ = {fitted_rates.max():.4f}")
