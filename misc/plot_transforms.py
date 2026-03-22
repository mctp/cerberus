import matplotlib.pyplot as plt
import numpy as np

# Define the range of x values (simulating signal track counts)
x = np.linspace(0, 10, 500)
x_large = np.linspace(0, 100, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Small range [0, 10]
ax1.plot(x, np.arcsinh(x), label="arcsinh(x)", linewidth=2)
ax1.plot(x, np.log1p(x), label="log(x+1)", linestyle="--", linewidth=2)
ax1.plot(x, np.sqrt(x), label="sqrt(x)", linestyle="-.", linewidth=2)
ax1.plot(x, np.tanh(x), label="tanh(x)", linestyle=":", linewidth=2)

ax1.set_title("Transformations (Small Range 0-10)")
ax1.set_xlabel("Input Signal (x)")
ax1.set_ylabel("Transformed Value")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Large range [0, 100] to show saturation vs growth
ax2.plot(x_large, np.arcsinh(x_large), label="arcsinh(x)", linewidth=2)
ax2.plot(x_large, np.log1p(x_large), label="log(x+1)", linestyle="--", linewidth=2)
ax2.plot(x_large, np.sqrt(x_large), label="sqrt(x)", linestyle="-.", linewidth=2)
ax2.plot(x_large, np.tanh(x_large), label="tanh(x)", linestyle=":", linewidth=2)

ax2.set_title("Transformations (Large Range 0-100)")
ax2.set_xlabel("Input Signal (x)")
ax2.set_ylabel("Transformed Value")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
output_path = "transform_comparison.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
