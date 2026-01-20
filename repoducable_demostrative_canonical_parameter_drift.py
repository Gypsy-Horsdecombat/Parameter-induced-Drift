#!/usr/bin/env python3
"""
canonical_parameter_drift.py

Minimal reproducible demonstration of parameter-induced drift.

An irreversible process with time-varying coupling κ(t) is generated.
A mis-specified exponential model is fitted over increasing observation
windows. The fitted decay rate λ̂(T) drifts systematically with window
length, demonstrating structural bias in rate-based descriptions.

Author: Gypsy Hors De Combat
License: CC0-1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Ground-truth irreversible dynamics
# -----------------------------

def kappa(t):
    """
    Time-dependent effective coupling κ(t).
    Slowly increasing (structured bath / senescence / memory effects).
    """
    return 0.15 + 0.08 * np.log(1.0 + t)

def order_functional(t):
    """
    O(t) = exp(-∫₀ᵗ κ(t') dt')
    Constructed numerically to avoid assuming exponential decay.
    """
    dt = t[1] - t[0]
    integral = np.cumsum(kappa(t)) * dt
    return np.exp(-integral)

# Time grid
t = np.linspace(0.0, 20.0, 4000)
O_true = order_functional(t)

# -----------------------------
# Mis-specified exponential model
# -----------------------------

def exp_model(t, lam):
    return np.exp(-lam * t)

# -----------------------------
# Sliding-window fits
# -----------------------------

window_lengths = np.linspace(2.0, 20.0, 30)
fitted_rates = []

for T in window_lengths:
    mask = t <= T
    t_fit = t[mask]
    O_fit = O_true[mask]

    # Fit exponential O ≈ exp(-λ t)
    popt, _ = curve_fit(exp_model, t_fit, O_fit, p0=[0.2])
    fitted_rates.append(popt[0])

fitted_rates = np.array(fitted_rates)

# -----------------------------
# Plotting
# -----------------------------

plt.figure(figsize=(13, 4))

# Panel A — true dynamics and representative exponential fits
plt.subplot(1, 2, 1)
plt.plot(t, O_true, 'k', lw=2.5, label='True irreversible dynamics')

for T in [4, 8, 12, 16, 20]:
    mask = t <= T
    popt, _ = curve_fit(exp_model, t[mask], O_true[mask], p0=[0.2])
    plt.plot(t, exp_model(t, popt[0]), '--', lw=1.5,
             label=f'Exp fit, T={T:.0f}, λ̂={popt[0]:.3f}')

plt.xlabel('t')
plt.ylabel('O(t)')
plt.title('Mis-specified exponential fits')
plt.legend(fontsize=8, frameon=False)

# Panel B — parameter-induced drift
plt.subplot(1, 2, 2)
plt.plot(window_lengths, fitted_rates, 'o-', lw=2.5)
plt.xlabel('Observation window T')
plt.ylabel('Fitted decay rate λ̂(T)')
plt.title('Parameter-induced drift')

# Highlight drift envelope
plt.axhline(fitted_rates.min(), ls=':', color='gray')
plt.axhline(fitted_rates.max(), ls=':', color='gray')

plt.tight_layout()
plt.show()

# -----------------------------
# Diagnostics (printable, quotable)
# -----------------------------

print("\nParameter-induced drift diagnostics\n")

for T, lam in zip(window_lengths[::5], fitted_rates[::5]):
    print(f"T = {T:6.2f}   λ̂(T) = {lam:.5f}")

print("\nDrift envelope:")
print(f"min λ̂ = {fitted_rates.min():.5f}")
print(f"max λ̂ = {fitted_rates.max():.5f}")
print(f"Δλ̂   = {fitted_rates.max() - fitted_rates.min():.5f}")

# Quantify monotonic trend
coef = np.polyfit(window_lengths, fitted_rates, 1)
print("\nLinear drift fit:")
print(f"dλ̂/dT ≈ {coef[0]:.5e}  (systematic parameter drift)")

print("\nInterpretation:")
print("Constant-rate exponential models absorb structural mismatch")
print("as drifting parameters. Rate is not primitive; it is a")
print("coordinate-dependent summary of reduced dynamics.\n")
