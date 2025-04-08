import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load the precalculated growth rates.
# This is the second output of the "calculate_growth_rate_at_contour" function:
with open("all_growth_rates.pkl", "rb") as f:
    all_growth_rates = pickle.load(f)

Ts = [-40, -20, 20, 80]

fig, axes = plt.subplots(4, 1, figsize=(6, 8))
axf = axes.flatten()

# Subsample the first two temperatures as they have longer videos
ss_1 = 5
ss_2 = 5
ss_3 = 1
ss_4 = 1

growth_rate_0, growth_rate_1, growth_rate_2, growth_rate_3 = all_growth_rates

t_offset = 0  # which time point to start from

# Plot the violin plots
vl = axf[0].violinplot(
    growth_rate_3[0::ss_4],
    positions=np.arange(t_offset, len(growth_rate_3) + t_offset)[0::ss_4],
    showextrema=False,
    widths=0.5 * ss_4,
)
axf[1].violinplot(
    growth_rate_2[0::ss_3],
    positions=np.arange(t_offset, len(growth_rate_2) + t_offset)[0::ss_3],
    showextrema=False,
    widths=0.5 * ss_3,
)
axf[2].violinplot(
    growth_rate_1[0::ss_2],
    positions=np.arange(t_offset, len(growth_rate_1) + t_offset)[0::ss_2],
    showextrema=False,
    widths=0.5 * ss_2,
)
axf[3].violinplot(
    growth_rate_0[0::ss_1],
    positions=np.arange(t_offset, len(growth_rate_0) + t_offset)[0::ss_1],
    showextrema=False,
    widths=0.5 * ss_1,
)


# Calculate the mean growth rate and plot on each akis
for i in range(4):
    growth_rates_i = all_growth_rates[i]
    mean_growth = [np.mean(growth_rate) for growth_rate in growth_rates_i]
    positions = np.arange(1, len(growth_rates_i) + 1)
    T_val = Ts[i]
    axf[3 - i].plot(mean_growth, color="b", label="T=%d" % T_val)

# Add a horizontal line at 0 for reference
for ax in axf:
    ax.axhline(0, color="k", linestyle="--")
    ax.legend()

# Miscellaneous formatting
plt.tight_layout()
axf[-1].set_xlabel("Time (s)")
plt.suptitle("Local Growth Rate at Different Temperatures over Time", x=0.55)
plt.tight_layout()
fig.subplots_adjust(left=0.15)
fig.text(
    0.04,
    0.5,
    r"Deposition rate ($\text{Intensity} / \text{nm } \text{s}$)",
    va="center",
    rotation="vertical",
)
plt.savefig("violin_plots.pdf")
plt.savefig("violin_plots.png")
plt.savefig("violin_plots.svg")
