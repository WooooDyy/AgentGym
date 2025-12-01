#!/usr/bin/env python3
"""
Visualize agent trajectories on a grid.

Creates ASCII visualization of agent paths through the state space.
"""

import json
from typing import List, Tuple


def parse_position(pos_str: str) -> Tuple[int, int]:
    """Parse position string like '(1, 2)' to tuple (1, 2)."""
    pos_str = pos_str.strip("()")
    x, y = pos_str.split(",")
    return int(x.strip()), int(y.strip())


def visualize_path(path: List[str], episode_id: int, actions: List[str]) -> str:
    """Create ASCII visualization of a path."""
    positions = [parse_position(p) for p in path]
    positions.insert(0, (0, 0))  # Add starting position

    # Find bounds
    all_x = [p[0] for p in positions]
    all_y = [p[1] for p in positions]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add padding
    min_x -= 1
    max_x += 1
    min_y -= 1
    max_y += 1

    # Create grid
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Plot path
    for i, (x, y) in enumerate(positions):
        grid_x = x - min_x
        grid_y = max_y - y  # Flip y-axis for display

        if i == 0:
            grid[grid_y][grid_x] = "S"  # Start
        elif i == len(positions) - 1:
            grid[grid_y][grid_x] = "E"  # End
        else:
            grid[grid_y][grid_x] = "●"  # Path point

    # Create visualization
    lines = []
    lines.append(f"\nEpisode {episode_id}: {' → '.join(actions)}")
    lines.append("─" * (width * 2 + 3))

    # Y-axis labels and grid
    for y in range(height):
        actual_y = max_y - y
        row_str = f"{actual_y:3d} │"
        for x in range(width):
            row_str += grid[y][x] + " "
        lines.append(row_str)

    # X-axis
    lines.append("    └" + "─" * (width * 2 - 1))
    x_labels = "     "
    for x in range(width):
        actual_x = min_x + x
        x_labels += (
            f"{actual_x:2d}"[0] if len(f"{actual_x:2d}") == 2 else f" {actual_x}"
        )
    lines.append(x_labels)

    # Legend
    lines.append("\nLegend: S=Start(0,0), ●=Path, E=End")

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Load trajectory data
    with open("trajectories.json", "r") as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print(" " * 22 + "TRAJECTORY VISUALIZATION")
    print("=" * 70)

    # Visualize each episode
    for episode in data["episodes"]:
        # Extract path from transitions
        path = []
        for trans in episode["transitions"]:
            obs = trans["observation"]
            if "position:" in obs:
                pos_str = obs.split("position:")[1].split(".")[0].strip()
                path.append(pos_str)

        actions = [t["action"] for t in episode["transitions"]]

        viz = visualize_path(path, episode["episode_id"], actions)
        print(viz)

    # Summary grid showing all final positions
    print("\n" + "=" * 70)
    print(" " * 25 + "ALL EPISODES SUMMARY")
    print("=" * 70)

    all_positions = [(0, 0)]  # Start position
    episode_labels = {(0, 0): "S"}

    for episode in data["episodes"]:
        transitions = episode["transitions"]
        if transitions:
            last_obs = transitions[-1]["observation"]
            if "position:" in last_obs:
                pos_str = last_obs.split("position:")[1].split(".")[0].strip()
                pos = parse_position(pos_str)
                all_positions.append(pos)
                episode_labels[pos] = str(episode["episode_id"])

    # Find bounds for summary
    all_x = [p[0] for p in all_positions]
    all_y = [p[1] for p in all_positions]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    min_x -= 1
    max_x += 1
    min_y -= 1
    max_y += 1

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Mark positions
    for pos, label in episode_labels.items():
        grid_x = pos[0] - min_x
        grid_y = max_y - pos[1]
        grid[grid_y][grid_x] = label

    print("\nFinal positions of all episodes:")
    print("─" * (width * 2 + 3))

    for y in range(height):
        actual_y = max_y - y
        row_str = f"{actual_y:3d} │"
        for x in range(width):
            row_str += grid[y][x] + " "
        print(row_str)

    print("    └" + "─" * (width * 2 - 1))
    x_labels = "     "
    for x in range(width):
        actual_x = min_x + x
        x_labels += (
            f"{actual_x:2d}"[0] if len(f"{actual_x:2d}") == 2 else f" {actual_x}"
        )
    print(x_labels)

    print("\nLegend: S=Start(0,0), Numbers=Episode final positions")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
