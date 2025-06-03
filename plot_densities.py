import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# Parameters
alpha = 70
x_k = 0.1
scale_param = 0.01
beta_values = [0.1, 1, 1.5, 2]
dashed_styles = ['--'] * len(beta_values)
dashed_colors = ['magenta', 'purple', 'brown', 'gray']
xi_values_finer = [-0.1, 0, 0.01, 0.05, 0.1]
colors_finer = ['blue', 'black', 'green', 'orange', 'red']
x_zoom = np.linspace(0, 0.5, 1000)

# Custom PDF
def fxk_pdf(x, alpha, beta, x_k):
    result = np.zeros_like(x)
    valid = x >= x_k
    result[valid] = alpha * beta * x[valid]**(beta - 1) * np.exp(-alpha * x[valid]**beta + alpha * x_k**beta)
    return result

# Custom Survival Function
def fxk_sf(x, alpha, beta, x_k):
    result = np.zeros_like(x)
    valid = x >= x_k
    result[valid] = np.exp(-alpha * x[valid]**beta + alpha * x_k**beta)
    return result

# Plot 1: Survival Function
plt.figure(figsize=(10, 6))
for xi, color in zip(xi_values_finer, colors_finer):
    y = genpareto.sf(x_zoom, xi, scale=scale_param, loc=x_k)
    label = f"GPD xi = {xi}" if xi != 0 else "GPD xi = 0 (Exponential)"
    plt.plot(x_zoom, y, label=label, color=color)

for beta, style, color in zip(beta_values, dashed_styles, dashed_colors):
    y = fxk_sf(x_zoom, alpha, beta, x_k)
    label = f"Custom Survival: $\\beta={beta}$"
    plt.plot(x_zoom, y, style, label=label, color=color)

plt.xlim(0, 0.5)
plt.yscale("log")
plt.ylim(1e-10, 1)
plt.xlabel("x")
plt.ylabel("Survival Function (log scale)")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Plot 2: Density Function
plt.figure(figsize=(10, 6))
for xi, color in zip(xi_values_finer, colors_finer):
    y = genpareto.pdf(x_zoom, xi, scale=scale_param, loc=x_k)
    label = f"GPD xi = {xi}" if xi != 0 else "GPD xi = 0 (Exponential)"
    plt.plot(x_zoom, y, label=label, color=color)

for beta, style, color in zip(beta_values, dashed_styles, dashed_colors):
    y = fxk_pdf(x_zoom, alpha, beta, x_k)
    label = f"Custom Density: $\\beta={beta}$"
    plt.plot(x_zoom, y, style, label=label, color=color)

plt.xlim(0, 0.5)
plt.ylim(0, 40)
plt.xlabel("x")
plt.ylabel("Density")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Density Function (log scale)
# Redraw the plot with no fill below x = 0.1

plt.figure(figsize=(10, 6))

# Plot GPD densities
for xi, color in zip(xi_values_finer, colors_finer):
    y = genpareto.pdf(x_zoom, xi, scale=scale_param, loc=x_k)
    label = f"GPD xi = {xi}" if xi != 0 else "GPD xi = 0 (Exponential)"
    plt.plot(x_zoom, y, label=label, color=color)

# Plot custom densities
for beta, style, color in zip(beta_values, dashed_styles, dashed_colors):
    y = fxk_pdf(x_zoom, alpha, beta, x_k)
    label = f"Custom Density: $\\beta={beta}$"
    plt.plot(x_zoom, y, style, label=label, color=color)

# Reference GPD xi = 0 line
y_xi0 = genpareto.pdf(x_zoom, 0, scale=scale_param, loc=x_k)

# Restrict filling to x >= x_k only
x_fill = x_zoom[x_zoom >= x_k]
y_fill = y_xi0[x_zoom >= x_k]

# Fill below and above only for x >= x_k
plt.fill_between(x_fill, 1e-10, y_fill, color='blue', alpha=0.08, label='Below GPD xi = 0')
plt.fill_between(x_fill, y_fill, 1e2, color='gold', alpha=0.08, label='Above GPD xi = 0')

# Final formatting
plt.xlim(0.09, 0.25)
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.xlabel("x")
plt.ylabel("Log Density")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True, which="major", ls="--")
plt.tight_layout()
plt.savefig("densities_log_new_scale.pdf")
plt.show()
