import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "results" / "random_agent_100_episodes.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"


def load_results():
    """Load episode results from CSV."""

    with CSV_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        results = list(reader)

    if not results:
        raise ValueError("The results CSV is empty.")

    return results


def save_figure(filename):
    """Save the current figure for publication."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_episode_outcomes(results):
    """Plot episode termination reasons."""

    outcomes = Counter()

    for result in results:
        if result["goal_reached"] == "True":
            outcomes["Goal reached"] += 1
        elif result["collision"] == "True":
            outcomes["Collision"] += 1
        elif result["out_of_bounds"] == "True":
            outcomes["Out of bounds"] += 1
        else:
            outcomes["Time limit"] += 1

    labels = [
        "Goal reached",
        "Collision",
        "Out of bounds",
        "Time limit",
    ]
    values = [outcomes[label] for label in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.title("Random Agent: Episode Outcomes")
    plt.ylabel("Number of Episodes")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            str(value),
            ha="center",
        )

    plt.ylim(0, max(values) * 1.15 if max(values) else 1)

    save_figure("random_agent_outcomes.png")


def plot_episode_lengths(results):
    """Plot the distribution of episode lengths."""

    steps = [int(result["steps"]) for result in results]

    plt.figure(figsize=(8, 5))
    plt.hist(steps, bins=15, edgecolor="black")

    plt.title("Random Agent: Episode Length Distribution")
    plt.xlabel("Steps")
    plt.ylabel("Number of Episodes")

    save_figure("random_agent_episode_lengths.png")


def plot_fuel_consumption(results):
    """Plot the distribution of fuel consumption."""

    fuel = [float(result["fuel_used"]) for result in results]

    plt.figure(figsize=(8, 5))
    plt.hist(fuel, bins=15, edgecolor="black")

    plt.title("Random Agent: Fuel Consumption Distribution")
    plt.xlabel("Fuel Used")
    plt.ylabel("Number of Episodes")

    save_figure("random_agent_fuel_consumption.png")


def main():
    results = load_results()

    plot_episode_outcomes(results)
    plot_episode_lengths(results)
    plot_fuel_consumption(results)

    print(f"Processed {len(results)} episodes.")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
