import torch

from experiments.lazy_pd.lazy_pd import (
    AdapterBank,
    Config,
    EncodedContext,
    Rollout,
    evidence_gate,
    make_stream,
)


def test_generation_stream_is_new_and_shifted():
    config = Config(generations=3, tasks_per_family=4)
    stream = make_stream(config)
    ids = [task.task_id for generation in stream for task in generation]
    assert len(ids) == len(set(ids))
    assert stream[0][0].correct_action != stream[1][0].correct_action


def test_evidence_gate_requires_success_and_multiple_tasks():
    config = Config(min_buffer_samples=2, min_buffer_tasks=2, min_signal=0.01)
    records = []
    for index in range(2):
        records.append(
            Rollout(
                task_id=f"task-{index}",
                family="x",
                adapter_key="x",
                adapter_version=0,
                prompt="p",
                privileged_prompts=("p1", "p2"),
                action=0,
                reward=float(index == 0),
                behavior_logp=-2.0,
                teacher_logp=-1.0,
                teacher_gain=1.0,
                teacher_consistency=0.1,
            )
        )
    passed, checks = evidence_gate(records, config)
    assert passed
    assert checks["verifier_gate"]
    records[0].reward = 0.0
    passed, _ = evidence_gate(records, config)
    assert not passed


def test_adapters_are_local_and_versioned():
    bank = AdapterBank(hidden_size=8, rank=2, device=torch.device("cpu"))
    context = EncodedContext(torch.ones(8), torch.zeros(4))
    before = bank.logits("family-a", context.hidden, context.base_logits)
    assert torch.allclose(before, torch.zeros(4))
    with torch.no_grad():
        bank.ensure("family-a").up.weight.fill_(1.0)
    after_a = bank.logits("family-a", context.hidden, context.base_logits)
    after_b = bank.logits("family-b", context.hidden, context.base_logits)
    assert not torch.allclose(after_a, after_b)
    assert bank.version("family-a") == 0
    bank.increment("family-a")
    assert bank.version("family-a") == 1
