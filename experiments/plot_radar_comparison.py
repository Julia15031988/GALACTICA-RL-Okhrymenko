from pathlib import Path

import matplotlib.pyplot as plt

# Results from the GALACTICA-RL radar experiment
# experiments/test_radar_16_vs_32.py
total_directions = 360

ray_counts = [16, 32, 72]
detected = [88, 184, 360]

percentages = [
    count / total_directions * 100
    for count in detected
]

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(
    [str(count) for count in ray_counts],
    percentages,
    color=["#64748b", "#38bdf8", "#8b5cf6"],
    width=0.55,
)

for bar, count, percentage in zip(
    bars, detected, percentages
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{percentage:.1f}%\n({count}/360)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_title(
    "GALACTICA-RL | Radar Detection Experiment",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

ax.set_xlabel("Number of Radar Rays")
ax.set_ylabel("Detection Rate (%)")

ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()

output = Path("results/radar_comparison.png")
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output, dpi=200, bbox_inches="tight")

print(f"Graph saved: {output.resolve()}")

plt.show()
