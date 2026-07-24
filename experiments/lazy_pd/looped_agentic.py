"""Long-loop self-distillation in a controlled multi-step agent environment.

The benchmark is deliberately procedural: every episode is a workflow that
requires a sequence of tool calls.  Common and rare workflow variants have
similar observations but different action programs.  A weak hindsight verifier
is combined with the current policy to form pseudo-labels, so an overconfident
wrong policy can overrule the verifier and reinforce its own error.

All methods start from the same warm-start policy and receive the same task
schedule and number of training examples:

* frozen: collect/evaluate, never update;
* eager: update after every wave, so new pseudo-labels depend immediately on
  every preceding update;
* lazy: keep the collector frozen for a complete cycle, pool all waves, then
  update once.

Confirmation bias is measured against the simulator's hidden oracle action.
Pattern loss is measured from per-pattern agent success histories.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


N_ACTIONS = 8


@dataclass
class Config:
    patterns: int = 24
    waves: int = 6
    cycles: int = 12
    horizon: int = 8
    attempts_per_step: int = 2
    tasks_per_pattern: int = 24
    eval_tasks_per_pattern: int = 32
    cue_dim: int = 24
    hidden_dim: int = 192
    verifier_weight: float = 0.22
    verifier_noise: float = 0.12
    label_temperature: float = 0.75
    learning_rate: float = 2e-3
    batch_size: int = 256
    warmup_epochs: int = 5
    update_epochs: int = 1
    confidence_threshold: float = 0.70
    ema_decay: float = 0.99
    replay_ratio: float = 1.0
    seed_patterns: int = 12
    seed_examples_per_pattern: int = 12
    methods: tuple[str, ...] = (
        "frozen", "eager", "eager_confident", "mean_teacher",
        "eager_replay", "lazy", "lazy_gated",
    )
    seed: int = 17
    device: str = "cuda"
    output_dir: str = "outputs/lazy_pd/looped_agentic"


@dataclass(frozen=True)
class Workflow:
    pattern: int
    cue: tuple[float, ...]
    program: tuple[int, ...]
    rare: bool
    pair: int


class Policy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_workflows(config: Config) -> list[Workflow]:
    """Create confusable common/rare pairs with conflicting action programs."""
    if config.patterns % 2:
        raise ValueError("--patterns must be even")
    rng = np.random.default_rng(config.seed)
    workflows = []
    for pair in range(config.patterns // 2):
        shared = rng.normal(size=config.cue_dim)
        shared /= np.linalg.norm(shared) + 1e-8
        common_program = rng.integers(0, N_ACTIONS, size=config.horizon)
        rare_program = common_program.copy()
        # Rare variants disagree at half the steps but have nearly identical
        # cues. This creates a real minority-pattern retention challenge.
        changed = rng.choice(
            config.horizon, size=max(2, config.horizon // 2), replace=False
        )
        rare_program[changed] = (
            rare_program[changed] + rng.integers(1, N_ACTIONS, size=len(changed))
        ) % N_ACTIONS
        for rare, program in ((False, common_program), (True, rare_program)):
            cue = shared + rng.normal(scale=0.10 if rare else 0.06, size=config.cue_dim)
            cue /= np.linalg.norm(cue) + 1e-8
            workflows.append(
                Workflow(
                    pattern=len(workflows),
                    cue=tuple(float(x) for x in cue),
                    program=tuple(int(x) for x in program),
                    rare=rare,
                    pair=pair,
                )
            )
    return workflows


def make_schedule(config: Config, workflows: list[Workflow]) -> list[list[int]]:
    """Each wave contains pairs, while later cycles increasingly favor common modes."""
    pairs = list(range(config.patterns // 2))
    random.Random(config.seed + 11).shuffle(pairs)
    wave_pairs = [pairs[i::config.waves] for i in range(config.waves)]
    return [
        [w.pattern for w in workflows if w.pair in set(group)]
        for group in wave_pairs
    ]


def observation(
    workflow: Workflow,
    progress: int,
    last_action: int,
    last_ok: bool,
    config: Config,
    noise_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    cue = np.asarray(workflow.cue) + rng.normal(scale=0.045, size=config.cue_dim)
    step = np.zeros(config.horizon, dtype=np.float32)
    step[progress] = 1.0
    previous = np.zeros(N_ACTIONS + 1, dtype=np.float32)
    previous[last_action if last_action >= 0 else N_ACTIONS] = 1.0
    feedback = np.asarray([float(last_ok), float(last_action >= 0 and not last_ok)])
    return np.concatenate([cue, step, previous, feedback]).astype(np.float32)


def dataset_batches(
    rows: list[tuple[np.ndarray, int, float]],
    batch_size: int,
    seed: int,
):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        selected = [rows[i] for i in order[start : start + batch_size]]
        yield (
            torch.from_numpy(np.stack([x[0] for x in selected])),
            torch.tensor([x[1] for x in selected], dtype=torch.long),
        )


def train(
    model: Policy,
    rows: list[tuple[np.ndarray, int, float]],
    config: Config,
    seed: int,
    epochs: int,
) -> dict:
    if not rows:
        return {"steps": 0, "examples": 0}
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses = []
    for epoch in range(epochs):
        for x, y in dataset_batches(rows, config.batch_size, seed + epoch):
            logits = model(x.to(config.device))
            loss = F.cross_entropy(logits, y.to(config.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return {
        "steps": len(losses),
        "examples": len(rows),
        "loss_start": losses[0],
        "loss_end": losses[-1],
    }


@torch.no_grad()
def policy_probs(model: Policy, obs: np.ndarray, config: Config) -> np.ndarray:
    x = torch.from_numpy(obs).to(config.device)
    if x.ndim == 1:
        x = x[None]
    return model(x).float().softmax(-1).cpu().numpy()


def pseudo_label(
    probs: np.ndarray,
    oracle: int,
    config: Config,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Weak privileged verifier plus self-policy belief.

    The verifier is correct most of the time, but its vote is intentionally too
    weak to repair an already confident wrong policy. This makes confirmation
    bias measurable rather than guaranteed: it occurs only when policy belief
    overwhelms valid hindsight.
    """
    verifier = np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float64)
    if rng.random() >= config.verifier_noise:
        verifier *= 0.15
        verifier[oracle] += 0.85
    else:
        wrong = int(rng.choice([x for x in range(N_ACTIONS) if x != oracle]))
        verifier *= 0.15
        verifier[wrong] += 0.85
    sharpened = np.power(np.maximum(probs, 1e-8), 1.0 / config.label_temperature)
    sharpened /= sharpened.sum()
    combined = (
        (1.0 - config.verifier_weight) * sharpened
        + config.verifier_weight * verifier
    )
    return int(combined.argmax()), float(combined.max())


def collect_wave(
    model: Policy,
    workflow_ids: list[int],
    workflows: list[Workflow],
    config: Config,
    cycle: int,
    wave: int,
) -> tuple[list[tuple[np.ndarray, int, float]], dict]:
    rows = []
    pseudo_correct = []
    self_reinforced = []
    confidences = []
    pseudo_actions = []
    oracle_actions = []
    episode_successes = []
    rare_flags = []
    rng = np.random.default_rng(config.seed + cycle * 1009 + wave * 97)
    for pattern in workflow_ids:
        workflow = workflows[pattern]
        # Later cycles oversample common patterns, reproducing a realistic
        # dominant-trajectory feedback stream without deleting rare patterns.
        multiplier = 3 if (cycle >= config.cycles // 3 and not workflow.rare) else 1
        episodes = config.tasks_per_pattern * multiplier
        progress = np.zeros(episodes, dtype=np.int64)
        last_action = np.full(episodes, -1, dtype=np.int64)
        last_ok = np.zeros(episodes, dtype=bool)
        active = np.ones(episodes, dtype=bool)
        budget = config.horizon * config.attempts_per_step
        for attempt in range(budget):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            observations = np.stack([
                observation(
                    workflow, int(progress[episode]), int(last_action[episode]),
                    bool(last_ok[episode]), config,
                    noise_seed=(
                        config.seed * 1_000_003 + cycle * 100_003 + wave * 10_007
                        + pattern * 503 + int(episode) * 31 + attempt
                    ),
                )
                for episode in indices
            ])
            probability_batch = policy_probs(model, observations, config)
            for local_index, episode in enumerate(indices):
                obs = observations[local_index]
                probs = probability_batch[local_index]
                behavior = int(rng.choice(N_ACTIONS, p=probs))
                oracle = workflow.program[int(progress[episode])]
                pseudo, confidence = pseudo_label(probs, oracle, config, rng)
                greedy = int(probs.argmax())
                rows.append((obs, pseudo, confidence))
                pseudo_correct.append(float(pseudo == oracle))
                self_reinforced.append(
                    float(greedy != oracle and pseudo == greedy)
                )
                confidences.append(confidence)
                pseudo_actions.append(pseudo)
                oracle_actions.append(oracle)
                rare_flags.append(workflow.rare)
                last_action[episode] = behavior
                last_ok[episode] = behavior == oracle
                if last_ok[episode]:
                    progress[episode] += 1
                    if progress[episode] == config.horizon:
                        active[episode] = False
        episode_successes.extend((progress == config.horizon).astype(float).tolist())
    counts = Counter(pseudo_actions)
    oracle_counts = Counter(oracle_actions)
    entropy = lambda c: -sum(
        (n / sum(c.values())) * math.log(max(n / sum(c.values()), 1e-12))
        for n in c.values()
    ) / math.log(N_ACTIONS)
    rare_mask = np.asarray(rare_flags, dtype=bool)
    correct_arr = np.asarray(pseudo_correct)
    metrics = {
        "examples": len(rows),
        "behavior_episode_success": float(np.mean(episode_successes)),
        "pseudo_accuracy": float(correct_arr.mean()),
        "rare_pseudo_accuracy": (
            float(correct_arr[rare_mask].mean()) if rare_mask.any() else None
        ),
        "common_pseudo_accuracy": (
            float(correct_arr[~rare_mask].mean()) if (~rare_mask).any() else None
        ),
        "self_reinforced_error_rate": float(np.mean(self_reinforced)),
        "mean_pseudo_confidence": float(np.mean(confidences)),
        "pseudo_action_entropy": float(entropy(counts)),
        "oracle_action_entropy": float(entropy(oracle_counts)),
        "dominant_pseudo_action_fraction": max(counts.values()) / len(pseudo_actions),
    }
    return rows, metrics


@torch.no_grad()
def evaluate(
    model: Policy,
    workflows: list[Workflow],
    config: Config,
    cycle: int,
    wave: int,
) -> dict:
    model.eval()
    episodes = config.eval_tasks_per_pattern
    shape = (len(workflows), episodes)
    progress = np.zeros(shape, dtype=np.int64)
    last_action = np.full(shape, -1, dtype=np.int64)
    last_ok = np.zeros(shape, dtype=bool)
    active = np.ones(shape, dtype=bool)
    step_correct: dict[int, list[float]] = defaultdict(list)
    for attempt in range(config.horizon * config.attempts_per_step):
        indices = np.argwhere(active)
        if not len(indices):
            break
        observations = np.stack([
            observation(
                workflows[int(pattern)], int(progress[pattern, episode]),
                int(last_action[pattern, episode]),
                bool(last_ok[pattern, episode]), config,
                noise_seed=(
                    900_000_011 + config.seed * 100_003 + int(pattern) * 997
                    + int(episode) * 37 + attempt
                ),
            )
            for pattern, episode in indices
        ])
        actions = policy_probs(model, observations, config).argmax(-1)
        for (pattern, episode), action in zip(indices, actions):
            workflow = workflows[int(pattern)]
            oracle = workflow.program[int(progress[pattern, episode])]
            ok = int(action) == oracle
            step_correct[int(pattern)].append(float(ok))
            last_action[pattern, episode] = int(action)
            last_ok[pattern, episode] = ok
            if ok:
                progress[pattern, episode] += 1
                if progress[pattern, episode] == config.horizon:
                    active[pattern, episode] = False
    per_pattern = {}
    for workflow in workflows:
        successes = progress[workflow.pattern] == config.horizon
        per_pattern[str(workflow.pattern)] = {
            "agent_success": float(np.mean(successes)),
            "step_accuracy": float(np.mean(step_correct[workflow.pattern])),
            "rare": workflow.rare,
        }
    values = list(per_pattern.values())
    rare = [x for x in values if x["rare"]]
    common = [x for x in values if not x["rare"]]
    return {
        "cycle": cycle,
        "wave": wave,
        "overall_agent_success": float(np.mean([x["agent_success"] for x in values])),
        "overall_step_accuracy": float(np.mean([x["step_accuracy"] for x in values])),
        "rare_agent_success": float(np.mean([x["agent_success"] for x in rare])),
        "common_agent_success": float(np.mean([x["agent_success"] for x in common])),
        "per_pattern": per_pattern,
    }


def warm_start(
    workflows: list[Workflow],
    config: Config,
) -> tuple[Policy, dict]:
    input_dim = config.cue_dim + config.horizon + N_ACTIONS + 1 + 2
    model = Policy(input_dim, config.hidden_dim).to(config.device)
    rng = random.Random(config.seed + 71)
    selected = list(range(config.patterns))
    rng.shuffle(selected)
    selected = selected[: config.seed_patterns]
    rows = []
    for pattern in selected:
        workflow = workflows[pattern]
        for example in range(config.seed_examples_per_pattern):
            progress = example % config.horizon
            # Cover the observation states that a long-horizon agent actually
            # visits after both successful and failed tool calls. Training only
            # the pristine first-attempt state creates an artificial horizon
            # floor because one early miss moves the policy out of distribution.
            state_rng = random.Random(
                config.seed * 100_003 + pattern * 997 + example
            )
            has_history = config.horizon > 8 and example >= config.horizon
            last_action = state_rng.randrange(N_ACTIONS) if has_history else -1
            last_ok = bool(state_rng.randrange(2)) if has_history else False
            obs = observation(
                workflow, progress, last_action, last_ok, config,
                noise_seed=config.seed * 701 + pattern * 41 + example,
            )
            rows.append((obs, workflow.program[progress], 1.0))
    update = train(
        model, rows, config, config.seed + 991, epochs=config.warmup_epochs
    )
    update["seed_patterns"] = selected
    return model, update


def pattern_loss(history: list[dict], schedule: list[list[int]]) -> dict:
    """Find retention loss only after a pattern has entered the data stream."""
    first_seen = {}
    for event, wave in enumerate(schedule):
        for pattern in wave:
            first_seen.setdefault(str(pattern), event)
    losses = {}
    for pattern, start in first_seen.items():
        scores = [
            row["evaluation"]["per_pattern"][pattern]["agent_success"]
            for row in history[start:]
        ]
        step_scores = [
            row["evaluation"]["per_pattern"][pattern]["step_accuracy"]
            for row in history[start:]
        ]
        losses[pattern] = {
            "peak_agent_success": max(scores),
            "final_agent_success": scores[-1],
            "agent_pattern_loss": max(scores) - scores[-1],
            "peak_step_accuracy": max(step_scores),
            "final_step_accuracy": step_scores[-1],
            "step_pattern_loss": max(step_scores) - step_scores[-1],
        }
    return {
        "mean_agent_pattern_loss": float(
            np.mean([x["agent_pattern_loss"] for x in losses.values()])
        ),
        "max_agent_pattern_loss": float(
            max(x["agent_pattern_loss"] for x in losses.values())
        ),
        "mean_step_pattern_loss": float(
            np.mean([x["step_pattern_loss"] for x in losses.values()])
        ),
        "patterns": losses,
    }


def run_method(
    method: str,
    initial_state: dict,
    workflows: list[Workflow],
    waves: list[list[int]],
    config: Config,
) -> dict:
    input_dim = config.cue_dim + config.horizon + N_ACTIONS + 1 + 2
    model = Policy(input_dim, config.hidden_dim).to(config.device)
    model.load_state_dict(initial_state)
    teacher = None
    if method == "mean_teacher":
        teacher = Policy(input_dim, config.hidden_dim).to(config.device)
        teacher.load_state_dict(initial_state)
        teacher.eval()
    history = []
    replay_buffer: list[tuple[np.ndarray, int, float]] = []
    replay_rng = random.Random(config.seed + 4049)
    for cycle in range(config.cycles):
        lazy_buffer = []
        for wave_id, pattern_ids in enumerate(waves):
            rows, collection = collect_wave(
                teacher if teacher is not None else model,
                pattern_ids, workflows, config, cycle, wave_id
            )
            update = None
            if method == "eager":
                update = train(
                    model, rows, config,
                    config.seed + cycle * config.waves + wave_id,
                    config.update_epochs,
                )
            elif method == "eager_confident":
                accepted = [
                    row for row in rows if row[2] >= config.confidence_threshold
                ]
                update = train(
                    model, accepted, config,
                    config.seed + cycle * config.waves + wave_id,
                    config.update_epochs,
                )
                update["accepted_fraction"] = len(accepted) / max(1, len(rows))
            elif method == "mean_teacher":
                update = train(
                    model, rows, config,
                    config.seed + cycle * config.waves + wave_id,
                    config.update_epochs,
                )
                assert teacher is not None
                with torch.no_grad():
                    for target, source in zip(teacher.parameters(), model.parameters()):
                        target.mul_(config.ema_decay).add_(
                            source, alpha=1.0 - config.ema_decay
                        )
            elif method == "eager_replay":
                replay_count = min(
                    len(replay_buffer), round(len(rows) * config.replay_ratio)
                )
                replay = (
                    replay_rng.sample(replay_buffer, replay_count)
                    if replay_count else []
                )
                update = train(
                    model, rows + replay, config,
                    config.seed + cycle * config.waves + wave_id,
                    config.update_epochs,
                )
                update["replayed"] = replay_count
                replay_buffer.extend(rows)
            elif method in {"lazy", "lazy_gated"}:
                lazy_buffer.extend(rows)
            if method in {"lazy", "lazy_gated"} and wave_id == len(waves) - 1:
                update_rows = lazy_buffer
                if method == "lazy_gated":
                    update_rows = [
                        row for row in lazy_buffer
                        if row[2] >= config.confidence_threshold
                    ]
                update = train(
                    model, update_rows, config,
                    config.seed + cycle * config.waves,
                    config.update_epochs,
                )
                if method == "lazy_gated":
                    update["accepted_fraction"] = (
                        len(update_rows) / max(1, len(lazy_buffer))
                    )
            evaluation = evaluate(
                model, workflows, config, cycle=cycle, wave=wave_id
            )
            history.append(
                {
                    "cycle": cycle,
                    "wave": wave_id,
                    "patterns": pattern_ids,
                    "collection": collection,
                    "update": update,
                    "evaluation": evaluation,
                }
            )
    confirmation = [x["collection"] for x in history]
    return {
        "method": method,
        "history": history,
        "confirmation_bias": {
            "mean_pseudo_error_rate": float(
                np.mean([1.0 - x["pseudo_accuracy"] for x in confirmation])
            ),
            "first_pseudo_error_rate": 1.0 - confirmation[0]["pseudo_accuracy"],
            "last_pseudo_error_rate": 1.0 - confirmation[-1]["pseudo_accuracy"],
            "error_amplification": (
                (1.0 - confirmation[-1]["pseudo_accuracy"])
                - (1.0 - confirmation[0]["pseudo_accuracy"])
            ),
            "mean_self_reinforced_error_rate": float(
                np.mean([x["self_reinforced_error_rate"] for x in confirmation])
            ),
            "final_self_reinforced_error_rate": confirmation[-1][
                "self_reinforced_error_rate"
            ],
            "mean_pseudo_action_entropy": float(
                np.mean([x["pseudo_action_entropy"] for x in confirmation])
            ),
        },
        "pattern_loss": pattern_loss(history, waves * config.cycles),
        "final": history[-1]["evaluation"],
    }


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    for name in (
        "patterns", "waves", "cycles", "horizon", "attempts_per_step",
        "tasks_per_pattern", "eval_tasks_per_pattern", "cue_dim", "hidden_dim",
        "batch_size", "warmup_epochs", "update_epochs", "seed_patterns",
        "seed_examples_per_pattern", "seed",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(Config, name)
        )
    for name in (
        "verifier_weight", "verifier_noise", "label_temperature",
        "learning_rate", "confidence_threshold", "ema_decay", "replay_ratio",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(Config, name)
        )
    parser.add_argument(
        "--methods", nargs="+",
        choices=(
            "frozen", "eager", "eager_confident", "mean_teacher",
            "eager_replay", "lazy", "lazy_gated",
        ),
        default=list(Config.methods),
    )
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--output-dir", default=Config.output_dir)
    args = parser.parse_args()
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)
    config.methods = tuple(args.methods)
    return config


def main() -> None:
    config = parse_args()
    if config.device == "cpu":
        torch.set_num_threads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.time()
    workflows = make_workflows(config)
    waves = make_schedule(config, workflows)
    initial, warmup = warm_start(workflows, config)
    initial_state = copy.deepcopy(initial.state_dict())
    initial_eval = evaluate(initial, workflows, config, cycle=-1, wave=-1)
    del initial
    results = [
        run_method(method, initial_state, workflows, waves, config)
        for method in config.methods
    ]
    summary = {
        row["method"]: {
            "final_agent_success": row["final"]["overall_agent_success"],
            "final_step_accuracy": row["final"]["overall_step_accuracy"],
            "final_rare_agent_success": row["final"]["rare_agent_success"],
            "confirmation_bias": row["confirmation_bias"],
            "mean_agent_pattern_loss": row["pattern_loss"][
                "mean_agent_pattern_loss"
            ],
            "mean_step_pattern_loss": row["pattern_loss"][
                "mean_step_pattern_loss"
            ],
        }
        for row in results
    }
    payload = {
        "config": asdict(config),
        "environment": {
            "actions": N_ACTIONS,
            "workflows": [asdict(x) for x in workflows],
            "waves": waves,
            "definition": {
                "confirmation_bias": (
                    "oracle pseudo-label error and the fraction of states where "
                    "a wrong greedy policy action becomes its own pseudo-label"
                ),
                "pattern_loss": (
                    "post-exposure peak minus final per-pattern agent success"
                ),
            },
        },
        "warmup": warmup,
        "initial": initial_eval,
        "summary": summary,
        "results": results,
        "hardware": {
            "device": config.device,
            "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
            "torch": torch.__version__,
            "runtime_seconds": time.time() - started,
        },
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"seed-{config.seed}.json"
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[looped-agentic] wrote {output_file}", flush=True)


if __name__ == "__main__":
    main()
