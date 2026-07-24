from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ACTIONS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    generation: int
    cue: str
    correct_action: int

    @property
    def prompt(self) -> str:
        return (
            "You control a scientific instrument. Select exactly one control action "
            "from A, B, C, or D. Explore when uncertain.\n"
            f"Instrument family: {self.family}\n"
            f"Opaque sample code: {self.cue}\n"
            "Action:"
        )


@dataclass
class EncodedContext:
    hidden: Tensor
    base_logits: Tensor


@dataclass
class Rollout:
    task_id: str
    family: str
    adapter_key: str
    adapter_version: int
    prompt: str
    privileged_prompts: tuple[str, str]
    action: int
    reward: float
    behavior_logp: float
    teacher_logp: float = 0.0
    teacher_gain: float = 0.0
    teacher_consistency: float = math.inf
    hidden: Tensor | None = None
    base_logits: Tensor | None = None


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    methods: tuple[str, ...] = ("frozen", "eager_global", "lazy_local")
    generations: int = 3
    tasks_per_family: int = 4
    families: tuple[str, ...] = ("conductivity", "heating", "mixing")
    rollouts_per_task: int = 8
    discovery_budget: int = 8
    temperature: float = 1.0
    lora_rank: int = 8
    learning_rate: float = 3e-3
    train_steps: int = 40
    max_grad_norm: float = 1.0
    importance_clip: float = 2.0
    min_buffer_samples: int = 16
    min_buffer_tasks: int = 2
    min_signal: float = 0.02
    max_teacher_std: float = 1.25
    base_branch_probability: float = 0.35
    seed: int = 17
    dtype: str = "bfloat16"
    device: str = "cuda"
    output_dir: str = "outputs/lazy_pd"


class ContextEncoder(Protocol):
    hidden_size: int

    def encode(self, prompts: Iterable[str]) -> dict[str, EncodedContext]: ...


class LoRAActionAdapter(nn.Module):
    """A local low-rank residual on the LM action logits."""

    def __init__(self, hidden_size: int, rank: int, num_actions: int = len(ACTIONS)):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, num_actions, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0 / rank

    def forward(self, hidden: Tensor) -> Tensor:
        return self.up(self.down(hidden)) * self.scale


class AdapterBank(nn.Module):
    def __init__(self, hidden_size: int, rank: int, device: torch.device):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.device = device
        self.adapters = nn.ModuleDict()
        self.versions: dict[str, int] = {}

    def ensure(self, key: str) -> LoRAActionAdapter:
        safe_key = key.replace("/", "__")
        if safe_key not in self.adapters:
            self.adapters[safe_key] = LoRAActionAdapter(
                self.hidden_size, self.rank
            ).to(self.device)
            self.versions[key] = 0
        return self.adapters[safe_key]

    def logits(
        self, key: str | None, hidden: Tensor, base_logits: Tensor
    ) -> Tensor:
        if key is None:
            return base_logits
        return base_logits + self.ensure(key)(hidden)

    def version(self, key: str) -> int:
        self.ensure(key)
        return self.versions[key]

    def increment(self, key: str) -> None:
        self.versions[key] = self.version(key) + 1


class HFContextEncoder:
    def __init__(self, model_name: str, device: str, dtype: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA validation was requested, but torch.cuda.is_available() is false"
            )
        self.device = torch.device(device)
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        self.action_token_ids = []
        for action in ACTIONS:
            ids = self.tokenizer.encode(" " + action, add_special_tokens=False)
            if not ids:
                raise ValueError(f"Tokenizer produced no ids for action {action}")
            self.action_token_ids.append(ids)

    @torch.no_grad()
    def encode(self, prompts: Iterable[str]) -> dict[str, EncodedContext]:
        unique = list(dict.fromkeys(prompts))
        result: dict[str, EncodedContext] = {}
        # Prompt counts are deliberately small. Processing one context at a time
        # avoids padding-position mistakes and makes candidate likelihood exact.
        for prompt in unique:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            action_scores = []
            hidden = None
            for action_ids in self.action_token_ids:
                ids = torch.tensor(
                    [prompt_ids + action_ids], device=self.device, dtype=torch.long
                )
                output = self.model(ids, output_hidden_states=True, use_cache=False)
                token_logp = F.log_softmax(output.logits.float(), dim=-1)
                score = torch.zeros((), device=self.device)
                for offset, token_id in enumerate(action_ids):
                    position = len(prompt_ids) + offset - 1
                    score = score + token_logp[0, position, token_id]
                action_scores.append(score)
                if hidden is None:
                    hidden = output.hidden_states[-1][0, len(prompt_ids) - 1].float()
            assert hidden is not None
            result[prompt] = EncodedContext(
                hidden=hidden.detach(),
                base_logits=torch.stack(action_scores).detach(),
            )
        return result


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def make_stream(config: Config) -> list[list[Task]]:
    """Make generation shifts that expose competence/evolvability tension."""
    stream: list[list[Task]] = []
    cue_words = (
        "amber", "birch", "cobalt", "delta", "ember", "flint", "garnet", "hazel",
        "indigo", "juniper", "kelp", "linen", "marble", "nickel", "onyx", "pearl",
    )
    for generation in range(config.generations):
        generation_tasks = []
        for family_index, family in enumerate(config.families):
            dominant = (generation + family_index) % len(ACTIONS)
            for index in range(config.tasks_per_family):
                cue_index = (
                    generation * config.tasks_per_family + index + family_index * 3
                ) % len(cue_words)
                # Three of four tasks share a generation-specific mode. The last
                # forces some support to remain on an alternative.
                correct = dominant if index < config.tasks_per_family - 1 else (
                    dominant + 1
                ) % len(ACTIONS)
                generation_tasks.append(
                    Task(
                        task_id=f"g{generation}/{family}/{index}",
                        family=family,
                        generation=generation,
                        cue=f"{cue_words[cue_index]}-{generation}-{index}",
                        correct_action=correct,
                    )
                )
        stream.append(generation_tasks)
    return stream


def privileged_prompts(task: Task, action: int, reward: float) -> tuple[str, str]:
    action_name = ACTIONS[action]
    if reward:
        observation = "The instrument activated and the verifier observed valid progress."
        verdict = "locally effective"
    else:
        observation = "The instrument did not change and the verifier observed no progress."
        verdict = "locally ineffective"
    first = (
        task.prompt
        + f"\nHindsight: control {action_name} was executed. Next observation: "
        + observation
        + f" The executed control was {verdict}.\n"
        "As a frozen verifier, score which control should be selected at the original state."
        "\nAction:"
    )
    second = (
        task.prompt
        + f"\nLocal transition record: attempted={action_name}; observation={observation}; "
        + f"verifier={verdict}. Judge the action for the pre-action state.\nAction:"
    )
    return first, second


def policy_probs(
    bank: AdapterBank,
    adapter_key: str | None,
    context: EncodedContext,
    temperature: float,
) -> Tensor:
    logits = bank.logits(adapter_key, context.hidden, context.base_logits)
    return F.softmax(logits / temperature, dim=-1)


def collect_task(
    task: Task,
    method: str,
    encoder: ContextEncoder,
    bank: AdapterBank,
    cache: dict[str, EncodedContext],
    config: Config,
    generator: torch.Generator,
) -> list[Rollout]:
    if task.prompt not in cache:
        cache.update(encoder.encode([task.prompt]))
    context = cache[task.prompt]
    records = []
    for _ in range(config.rollouts_per_task):
        use_base = method == "frozen" or (
            method == "lazy_local"
            and torch.rand((), generator=generator).item()
            < config.base_branch_probability
        )
        key = None if use_base else (
            "global" if "global" in method else task.family
        )
        probs = policy_probs(bank, key, context, config.temperature)
        action = int(torch.multinomial(probs.cpu(), 1, generator=generator).item())
        reward = float(action == task.correct_action)
        behavior_logp = float(torch.log(probs[action].clamp_min(1e-12)).item())
        teacher_key = key
        version_key = "base" if teacher_key is None else teacher_key
        version = 0 if teacher_key is None else bank.version(teacher_key)
        records.append(
            Rollout(
                task_id=task.task_id,
                family=task.family,
                adapter_key=version_key,
                adapter_version=version,
                prompt=task.prompt,
                privileged_prompts=privileged_prompts(task, action, reward),
                action=action,
                reward=reward,
                behavior_logp=behavior_logp,
                hidden=context.hidden.detach().cpu(),
                base_logits=context.base_logits.detach().cpu(),
            )
        )
    return records


@torch.no_grad()
def teacher_score(
    records: list[Rollout],
    encoder: ContextEncoder,
    bank: AdapterBank,
    config: Config,
) -> None:
    prompts = [p for record in records for p in record.privileged_prompts]
    contexts = encoder.encode(prompts)
    for record in records:
        logps = []
        teacher_key = None if record.adapter_key == "base" else record.adapter_key
        for prompt in record.privileged_prompts:
            context = contexts[prompt]
            logits = bank.logits(
                teacher_key, context.hidden, context.base_logits
            )
            logps.append(
                F.log_softmax(logits / config.temperature, dim=-1)[record.action]
            )
        stacked = torch.stack(logps)
        record.teacher_logp = float(stacked.mean().item())
        record.teacher_gain = record.teacher_logp - record.behavior_logp
        record.teacher_consistency = float(stacked.std(unbiased=False).item())


def evidence_gate(records: list[Rollout], config: Config) -> tuple[bool, dict]:
    tasks = {record.task_id for record in records}
    successful_tasks = {record.task_id for record in records if record.reward > 0}
    consistent = [
        record for record in records
        if record.teacher_consistency <= config.max_teacher_std
    ]
    mean_signal = (
        float(np.mean([abs(r.teacher_gain) for r in consistent]))
        if consistent else 0.0
    )
    checks = {
        "samples": len(records),
        "independent_tasks": len(tasks),
        "successful_tasks": len(successful_tasks),
        "consistent_fraction": len(consistent) / max(1, len(records)),
        "mean_abs_signal": mean_signal,
        "sample_gate": len(records) >= config.min_buffer_samples,
        "task_gate": len(tasks) >= config.min_buffer_tasks,
        "verifier_gate": bool(successful_tasks),
        "signal_gate": mean_signal >= config.min_signal,
        "consistency_gate": len(consistent) >= max(1, len(records) // 2),
    }
    passed = all(
        checks[name]
        for name in (
            "sample_gate", "task_gate", "verifier_gate",
            "signal_gate", "consistency_gate",
        )
    )
    return passed, checks


def distill(
    records: list[Rollout],
    adapter_key: str,
    bank: AdapterBank,
    config: Config,
) -> dict:
    # A successful peer is required at task level. Failed actions from that same
    # task remain useful because local hindsight tells the teacher they failed.
    successful_tasks = {record.task_id for record in records if record.reward > 0}
    filtered = [
        record for record in records
        if record.task_id in successful_tasks
        and record.teacher_consistency <= config.max_teacher_std
        and record.adapter_key == adapter_key
        and record.adapter_version == bank.version(adapter_key)
    ]
    if not filtered:
        return {"updated": False, "reason": "no version-consistent scored samples"}
    adapter = bank.ensure(adapter_key)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate)
    hidden = torch.stack([r.hidden for r in filtered]).to(bank.device)
    base_logits = torch.stack([r.base_logits for r in filtered]).to(bank.device)
    actions = torch.tensor([r.action for r in filtered], device=bank.device)
    behavior = torch.tensor(
        [r.behavior_logp for r in filtered], device=bank.device
    )
    teacher = torch.tensor(
        [r.teacher_logp for r in filtered], device=bank.device
    )
    losses = []
    for _ in range(config.train_steps):
        current_all = F.log_softmax(
            bank.logits(adapter_key, hidden, base_logits) / config.temperature,
            dim=-1,
        )
        current = current_all.gather(1, actions[:, None]).squeeze(1)
        ratio = torch.exp(current - behavior).clamp(
            max=config.importance_clip
        )
        # Monte-Carlo reverse-KL: E_student[log pi_s - log pi_t].
        # The score-function advantage is detached, while ratio provides the
        # current-policy gradient and corrects bounded behavior staleness.
        advantage = (current - teacher).detach()
        loss = (ratio * advantage).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), config.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    bank.increment(adapter_key)
    return {
        "updated": True,
        "samples": len(filtered),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "new_version": bank.version(adapter_key),
    }


def task_distribution(
    task: Task,
    method: str,
    encoder: ContextEncoder,
    bank: AdapterBank,
    cache: dict[str, EncodedContext],
    config: Config,
) -> Tensor:
    if task.prompt not in cache:
        cache.update(encoder.encode([task.prompt]))
    context = cache[task.prompt]
    if method == "frozen":
        return policy_probs(bank, None, context, config.temperature)
    key = "global" if "global" in method else task.family
    adapted = policy_probs(bank, key, context, config.temperature)
    if method == "lazy_local":
        base = policy_probs(bank, None, context, config.temperature)
        return (
            config.base_branch_probability * base
            + (1.0 - config.base_branch_probability) * adapted
        )
    return adapted


def evaluate(
    tasks: list[Task],
    method: str,
    encoder: ContextEncoder,
    bank: AdapterBank,
    cache: dict[str, EncodedContext],
    config: Config,
) -> dict:
    correct_probs, greedy, discovery, entropies = [], [], [], []
    action_mass = torch.zeros(len(ACTIONS))
    for task in tasks:
        probs = task_distribution(task, method, encoder, bank, cache, config).cpu()
        p = float(probs[task.correct_action].item())
        correct_probs.append(p)
        greedy.append(float(int(probs.argmax()) == task.correct_action))
        discovery.append(1.0 - (1.0 - p) ** config.discovery_budget)
        entropies.append(
            float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
            / math.log(len(ACTIONS))
        )
        action_mass += probs
    action_mass /= max(1, len(tasks))
    return {
        "n_tasks": len(tasks),
        "success_at_1_expected": float(np.mean(correct_probs)),
        "success_at_1_greedy": float(np.mean(greedy)),
        "discovery_at_b_expected": float(np.mean(discovery)),
        "normalized_action_entropy": float(np.mean(entropies)),
        "mean_action_mass": [float(x) for x in action_mass],
    }


def run_method(
    method: str,
    encoder: ContextEncoder,
    stream: list[list[Task]],
    config: Config,
) -> dict:
    device = torch.device(config.device)
    bank = AdapterBank(encoder.hidden_size, config.lora_rank, device)
    cache: dict[str, EncodedContext] = {}
    buffers: dict[str, list[Rollout]] = defaultdict(list)
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + stable_int(method) % 100_000
    )
    generations = []
    for generation_index, current_tasks in enumerate(stream):
        future_tasks = [
            task for tasks in stream[generation_index + 1 :] for task in tasks
        ]
        past_tasks = [task for tasks in stream[:generation_index] for task in tasks]
        pre_future = (
            evaluate(future_tasks, method, encoder, bank, cache, config)
            if future_tasks else None
        )
        updates, gates = [], []
        if method != "frozen":
            for task in current_tasks:
                records = collect_task(
                    task, method, encoder, bank, cache, config, generator
                )
                teacher_score(records, encoder, bank, config)
                key = "global" if "global" in method else task.family
                # Base-branch rollouts preserve exploration but do not train an
                # adapter: only on-policy rollouts from the adapter version do.
                adapter_records = [r for r in records if r.adapter_key == key]
                buffers[key].extend(adapter_records)
                should_check = method == "eager_global" or method.startswith(
                    "immediate"
                )
                if should_check:
                    passed, gate = evidence_gate(buffers[key], config)
                    gates.append({"key": key, "task": task.task_id, **gate})
                    if passed:
                        updates.append(distill(buffers[key], key, bank, config))
                        buffers[key].clear()
            if method in {
                "delayed_global", "lazy_local_no_base", "lazy_local"
            }:
                for key in list(buffers):
                    passed, gate = evidence_gate(buffers[key], config)
                    gates.append({"key": key, "task": "generation_end", **gate})
                    if passed:
                        updates.append(distill(buffers[key], key, bank, config))
                        buffers[key].clear()
        current_eval = evaluate(
            current_tasks, method, encoder, bank, cache, config
        )
        past_eval = (
            evaluate(past_tasks, method, encoder, bank, cache, config)
            if past_tasks else None
        )
        future_eval = (
            evaluate(future_tasks, method, encoder, bank, cache, config)
            if future_tasks else None
        )
        generations.append(
            {
                "generation": generation_index,
                "adapter_versions": copy.deepcopy(bank.versions),
                "updates": updates,
                "gates": gates,
                "current": current_eval,
                "past": past_eval,
                "future_before_update": pre_future,
                "future_after_update": future_eval,
            }
        )
    return {"method": method, "generations": generations}


def aggregate_claim(results: list[dict]) -> dict:
    summary = {}
    for result in results:
        rows = result["generations"]
        current = [row["current"]["success_at_1_expected"] for row in rows]
        future = [
            row["future_after_update"]["discovery_at_b_expected"]
            for row in rows if row["future_after_update"] is not None
        ]
        entropy = [
            row["future_after_update"]["normalized_action_entropy"]
            for row in rows if row["future_after_update"] is not None
        ]
        summary[result["method"]] = {
            "mean_current_success_at_1": float(np.mean(current)),
            "mean_future_discovery_at_b": float(np.mean(future)) if future else None,
            "mean_future_action_entropy": float(np.mean(entropy)) if entropy else None,
            "total_updates": sum(
                len(row["updates"]) for row in result["generations"]
            ),
        }
    return summary


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Config.model)
    parser.add_argument(
        "--methods", nargs="+", default=list(Config.methods),
        choices=(
            "frozen", "eager_global", "delayed_global", "immediate_local",
            "lazy_local_no_base", "lazy_local",
        ),
    )
    parser.add_argument("--generations", type=int, default=Config.generations)
    parser.add_argument(
        "--tasks-per-family", type=int, default=Config.tasks_per_family
    )
    parser.add_argument(
        "--rollouts-per-task", type=int, default=Config.rollouts_per_task
    )
    parser.add_argument("--train-steps", type=int, default=Config.train_steps)
    parser.add_argument(
        "--base-branch-probability",
        type=float,
        default=Config.base_branch_probability,
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--dtype", default=Config.dtype)
    parser.add_argument("--output-dir", default=Config.output_dir)
    args = parser.parse_args()
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key.replace("-", "_"), value)
    config.methods = tuple(args.methods)
    return config


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    started = time.time()
    encoder = HFContextEncoder(config.model, config.device, config.dtype)
    stream = make_stream(config)
    results = []
    for method in config.methods:
        print(f"[lazy-pd] running {method}", flush=True)
        results.append(run_method(method, encoder, stream, config))
    payload = {
        "config": asdict(config),
        "hardware": {
            "device": str(encoder.device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if encoder.device.type == "cuda" else None
            ),
            "torch": torch.__version__,
        },
        "runtime_seconds": time.time() - started,
        "summary": aggregate_claim(results),
        "results": results,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"[lazy-pd] wrote {output_file}", flush=True)


if __name__ == "__main__":
    main()
