import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Grip distribution configuration
grip_distribution_types = {
    "wide_high": {
        "force": {"mu": 191, "sigma": 1},
        "color": "#E6C619",   # yellow
        "label": "Wide High (μ=191, σ=2)"
    },
    "narrow_high": {
        "force": {"mu": 191, "sigma": 0.5},
        "color": "#D94F3D",   # red
        "label": "Narrow High (μ=191, σ=0.7)"
    },
    "wide_low": {
        "force": {"mu": 188, "sigma": 1},
        "color": "#4CAF82",   # green
        "label": "Wide Low (μ=189, σ=2)"
    },
    "narrow_low": {
        "force": {"mu": 188, "sigma": 0.5},
        "color": "#4A90D9",   # blue
        "label": "Narrow Low (μ=189, σ=0.7)"
    }
}

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(182, 200, 1000)

for key, config in grip_distribution_types.items():
    mu = config["force"]["mu"]
    sigma = config["force"]["sigma"]
    color = config["color"]
    label = config["label"]

    y = gaussian(x, mu, sigma)

    # Solid line
    ax.plot(x, y, color=color, linewidth=2.5, label=label)

    # Shaded area under curve
    ax.fill_between(x, y, alpha=0.12, color=color)

    # Mark the mean with a vertical dashed line
    ax.axvline(mu, color=color, linestyle="--", linewidth=1.0, alpha=0.6)

# --- 90% accuracy threshold ---
#ax.axvline(190, color="black", linestyle=":", linewidth=1.5, alpha=0.5, label="90% accuracy threshold (force=190)")

# --- Annotations ---
ax.set_title("Claw Grip Force Distributions", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Force (arbitrary units)", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.set_xlim(182, 200)
ax.grid(True, linestyle="--", alpha=0.35)

# Light grey background
ax.set_facecolor("#F8F8F8")
fig.patch.set_facecolor("#FFFFFF")

plt.tight_layout()
plt.savefig("claw_force_distributions.png", dpi=150)
plt.show()
print("Plot saved as claw_force_distributions.png")