"""Create the paper's long-horizon dynamics figure from saved JSON runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "outputs/lazy_pd/long_horizon_evolution"
SDAR = ROOT / "outputs/lazy_pd/sdar_style_long"
OUT = ROOT / "reports/figures"

COLORS = {
    "Eager": "#D55E00",
    "SDAR-style": "#CC79A7",
    "Mean Teacher": "#0072B2",
    "Lazy": "#009E73",
}


def load_runs(path: Path, method: str) -> list[dict]:
    runs = []
    for filename in sorted(path.glob("seed-*.json")):
        payload = json.loads(filename.read_text())
        result = next(row for row in payload["results"] if row["method"] == method)
        runs.append(result)
    return runs


def cycle_series(run: dict, key: str) -> np.ndarray:
    """Use the last evaluation/collection event in each six-wave cycle."""
    history = run["history"]
    cycles = sorted({row["cycle"] for row in history})
    values = []
    for cycle in cycles:
        row = [x for x in history if x["cycle"] == cycle][-1]
        if key == "success":
            values.append(row["evaluation"]["overall_agent_success"])
        elif key == "pseudo_error":
            values.append(1.0 - row["collection"]["pseudo_accuracy"])
        else:
            raise ValueError(key)
    return np.asarray(values)


def mean_band(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([cycle_series(run, key) for run in runs])
    return values.mean(0), values.std(0, ddof=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    methods = {
        "Eager": load_runs(MAIN, "eager"),
        "SDAR-style": load_runs(SDAR, "sdar_style"),
        "Mean Teacher": load_runs(MAIN, "mean_teacher"),
        "Lazy": load_runs(MAIN, "lazy"),
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.75))
    x = np.arange(1, 25)

    for name, runs in methods.items():
        for axis, key in zip(axes[:2], ("success", "pseudo_error")):
            mean, std = mean_band(runs, key)
            axis.plot(x, 100 * mean, label=name, color=COLORS[name], linewidth=2)
            axis.fill_between(
                x, 100 * (mean - std), 100 * (mean + std),
                color=COLORS[name], alpha=0.12, linewidth=0,
            )

    axes[0].set_title("(a) Task ability")
    axes[0].set_ylabel("Full-episode success (%)")
    axes[1].set_title("(b) Confirmation bias")
    axes[1].set_ylabel("Pseudo-label error (%)")
    for axis in axes[:2]:
        axis.set_xlabel("Self-evolution cycle")
        axis.set_xlim(1, 24)
        axis.set_ylim(-3, 103)
        axis.grid(axis="y", alpha=0.22)

    names = list(methods)
    losses = []
    errors = []
    for name, runs in methods.items():
        values = [100 * r["pattern_loss"]["mean_agent_pattern_loss"] for r in runs]
        losses.append(np.mean(values))
        errors.append(np.std(values, ddof=1))
    positions = np.arange(len(names))
    axes[2].bar(
        positions, losses, yerr=errors, capsize=3,
        color=[COLORS[name] for name in names], alpha=0.9,
    )
    axes[2].set_title("(c) Pattern loss")
    axes[2].set_ylabel("Peak-to-final loss (%)")
    axes[2].set_xticks(positions, ["Eager", "SDAR-\nstyle", "Mean\nTeacher", "Lazy"])
    axes[2].set_ylim(0, 62)
    axes[2].grid(axis="y", alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.4)
    fig.savefig(OUT / "long_horizon_dynamics.pdf", bbox_inches="tight")
    fig.savefig(OUT / "long_horizon_dynamics.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
