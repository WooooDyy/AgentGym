#!/usr/bin/env python3
"""
Analyze trajectory data and generate evaluation metrics.

This script loads trajectory data and computes various metrics:
- Reward statistics per episode
- Action distribution
- State visitation patterns
- Success metrics
"""

import json
from collections import Counter
from typing import Dict, Any
import sys


def load_trajectories(filepath: str = "trajectories.json") -> Dict[str, Any]:
    """Load trajectory data from JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        sys.exit(1)


def analyze_actions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze action distribution across all episodes."""
    all_actions = []
    for episode in data["episodes"]:
        for transition in episode["transitions"]:
            all_actions.append(transition["action"])

    action_counts = Counter(all_actions)
    total_actions = len(all_actions)

    return {
        "action_counts": dict(action_counts),
        "action_frequencies": {
            action: count / total_actions for action, count in action_counts.items()
        },
        "total_actions": total_actions,
        "unique_actions": len(action_counts),
    }


def analyze_states(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze state visitation patterns."""
    states_visited = []
    final_states = []

    for episode in data["episodes"]:
        episode_states = []
        for transition in episode["transitions"]:
            # Extract position from observation
            obs = transition["observation"]
            if "position:" in obs:
                # Parse position (x, y) from observation
                pos_str = obs.split("position:")[1].split(".")[0].strip()
                episode_states.append(pos_str)

        states_visited.extend(episode_states)
        if episode_states:
            final_states.append(episode_states[-1])

    state_counts = Counter(states_visited)
    final_state_counts = Counter(final_states)

    return {
        "unique_states_visited": len(state_counts),
        "most_common_states": state_counts.most_common(5),
        "final_states": dict(final_state_counts),
        "total_state_visits": len(states_visited),
    }


def analyze_rewards(data: Dict[str, Any]) -> Dict[str, Any]:
    """Detailed reward analysis."""
    episode_rewards = []
    step_rewards = []

    for episode in data["episodes"]:
        episode_rewards.append(episode["total_reward"])
        for transition in episode["transitions"]:
            step_rewards.append(transition["reward"])

    return {
        "per_episode": {
            "mean": sum(episode_rewards) / len(episode_rewards),
            "std": (
                sum(
                    (r - sum(episode_rewards) / len(episode_rewards)) ** 2
                    for r in episode_rewards
                )
                / len(episode_rewards)
            )
            ** 0.5,
            "min": min(episode_rewards),
            "max": max(episode_rewards),
            "all_rewards": episode_rewards,
        },
        "per_step": {
            "mean": sum(step_rewards) / len(step_rewards),
            "min": min(step_rewards),
            "max": max(step_rewards),
            "all_rewards": step_rewards,
        },
    }


def compute_episode_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics for a single episode."""
    transitions = episode["transitions"]

    # Efficiency: reward per step
    efficiency = (
        episode["total_reward"] / episode["total_steps"]
        if episode["total_steps"] > 0
        else 0
    )

    # Extract trajectory path
    path = []
    for trans in transitions:
        obs = trans["observation"]
        if "position:" in obs:
            pos_str = obs.split("position:")[1].split(".")[0].strip()
            path.append(pos_str)

    return {
        "episode_id": episode["episode_id"],
        "total_reward": episode["total_reward"],
        "total_steps": episode["total_steps"],
        "efficiency": efficiency,
        "success": episode["success"],
        "path": path,
        "actions": [t["action"] for t in transitions],
    }


def generate_report(data: Dict[str, Any]):
    """Generate comprehensive analysis report."""
    print("\n" + "=" * 70)
    print(" " * 20 + "TRAJECTORY ANALYSIS REPORT")
    print("=" * 70)

    # Overall statistics
    stats = data["statistics"]
    print(f"\n{'OVERALL STATISTICS':-^70}")
    print(f"  Total Episodes:        {stats['num_episodes']}")
    print(f"  Total Transitions:     {stats['total_transitions']}")
    print(f"  Success Rate:          {stats['success_rate']:.1%}")
    print(f"  Average Steps/Episode: {stats['avg_steps']:.2f}")

    # Reward analysis
    reward_analysis = analyze_rewards(data)
    print(f"\n{'REWARD ANALYSIS':-^70}")
    print("  Per Episode:")
    print(f"    Mean:   {reward_analysis['per_episode']['mean']:.4f}")
    print(f"    Std:    {reward_analysis['per_episode']['std']:.4f}")
    print(f"    Min:    {reward_analysis['per_episode']['min']:.4f}")
    print(f"    Max:    {reward_analysis['per_episode']['max']:.4f}")
    print("  Per Step:")
    print(f"    Mean:   {reward_analysis['per_step']['mean']:.4f}")
    print(f"    Min:    {reward_analysis['per_step']['min']:.4f}")
    print(f"    Max:    {reward_analysis['per_step']['max']:.4f}")

    # Action analysis
    action_analysis = analyze_actions(data)
    print(f"\n{'ACTION DISTRIBUTION':-^70}")
    print(f"  Total Actions:  {action_analysis['total_actions']}")
    print(f"  Unique Actions: {action_analysis['unique_actions']}")
    print("  Frequencies:")
    for action, freq in sorted(
        action_analysis["action_frequencies"].items(), key=lambda x: x[1], reverse=True
    ):
        count = action_analysis["action_counts"][action]
        print(f"    {action:12s}: {count:3d} ({freq:.1%})")

    # State analysis
    state_analysis = analyze_states(data)
    print(f"\n{'STATE ANALYSIS':-^70}")
    print(f"  Unique States Visited: {state_analysis['unique_states_visited']}")
    print(f"  Total State Visits:    {state_analysis['total_state_visits']}")
    print("  Most Common States:")
    for state, count in state_analysis["most_common_states"]:
        print(f"    {state:20s}: {count:3d} visits")
    print("  Final States:")
    for state, count in state_analysis["final_states"].items():
        print(f"    {state:20s}: {count:3d} episode(s)")

    # Per-episode metrics
    print(f"\n{'PER-EPISODE METRICS':-^70}")
    print(
        f"  {'Ep':>3} | {'Steps':>5} | {'Reward':>7} | {'Efficiency':>10} | {'Success':>7}"
    )
    print(f"  {'-' * 3}-+-{'-' * 5}-+-{'-' * 7}-+-{'-' * 10}-+-{'-' * 7}")

    for episode in data["episodes"]:
        metrics = compute_episode_metrics(episode)
        print(
            f"  {metrics['episode_id']:3d} | "
            f"{metrics['total_steps']:5d} | "
            f"{metrics['total_reward']:7.3f} | "
            f"{metrics['efficiency']:10.4f} | "
            f"{'✓' if metrics['success'] else '✗':>7}"
        )

    # Episode details
    print(f"\n{'EPISODE DETAILS':-^70}")
    for episode in data["episodes"]:
        metrics = compute_episode_metrics(episode)
        print(f"\n  Episode {metrics['episode_id']}:")
        print(f"    Actions: {' → '.join(metrics['actions'])}")
        print(f"    Path:    {' → '.join(metrics['path'])}")
        print(
            f"    Reward:  {metrics['total_reward']:.3f} (efficiency: {metrics['efficiency']:.4f})"
        )

    print(f"\n{'=' * 70}\n")


def main():
    """Main entry point."""
    # Load data
    print("Loading trajectory data...")
    data = load_trajectories("trajectories.json")

    # Generate report
    generate_report(data)

    # Export summary
    summary = {
        "overall_stats": data["statistics"],
        "reward_analysis": analyze_rewards(data),
        "action_analysis": analyze_actions(data),
        "state_analysis": analyze_states(data),
        "episode_metrics": [compute_episode_metrics(ep) for ep in data["episodes"]],
    }

    with open("trajectory_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("📊 Detailed analysis saved to trajectory_analysis.json")


if __name__ == "__main__":
    main()
