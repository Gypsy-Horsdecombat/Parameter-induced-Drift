import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Time-varying coupling (the real process)
def kappa(t):
    return 0.2 + 0.2*np.sin(0.3*t) + 0.05*t/(1+t)

# Integral of kappa → true irreversible process
def order_functional(t):
    xs = []
    for ti in t:
        ts = np.linspace(0, ti, 1000)
        xs.append(np.trapz(kappa(ts), ts))
    return np.exp(-np.array(xs))

# Wrong model (what people usually fit)
def exp_model(t, lam, O0):
    return O0 * np.exp(-lam * t)

# Fit exponential over window [0, T]
def fit_lambda(T):
    t = np.linspace(0, T, 400)
    O = order_functional(t)
    popt, _ = curve_fit(exp_model, t, O, p0=(0.5, 1.0))
    return popt[0]

# Main demo
Tvals = np.linspace(2, 40, 25)
lams = [fit_lambda(T) for T in Tvals]

plt.plot(Tvals, lams, "o-")
plt.xlabel("Observation window T")
plt.ylabel("Fitted decay rate λ(T)")
plt.title("Parameter-induced drift from mis-specified exponential fit")
plt.show()
