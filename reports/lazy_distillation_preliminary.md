# Lazy Distillation for Long-Horizon Self-Evolving Agents

Preliminary controlled evidence, July 2026.

## Core claim

Confirmation bias and pattern loss are failures of a repeated, long-horizon
self-evolution loop:

```text
policy → multi-step trajectories → pseudo-labels → policy update
       ↑_______________________________________________|
```

An early error can affect which states are visited, become its own
pseudo-label, change the policy, and contaminate later experience. Lazy
Distillation freezes the trajectory collector for a multi-wave window and
updates once on the pooled evidence.

- Confirmation bias is reduced because an update cannot contaminate later
  trajectories inside the same collection window.
- Pattern loss is reduced because common and rare workflow patterns are present
  together in the update instead of sequentially overwriting one another.

## Controlled benchmark

- 24 confusable common/rare workflow patterns
- 8 dependent tool actions per episode
- 6 non-IID waves × 12 self-evolution cycles
- 5 random seeds
- Same initialization and task schedule for every method

We measure confirmation bias using simulator oracle actions. A
`self-reinforced error` occurs when the policy's greedy action is wrong and its
pseudo-label repeats that same wrong action. Pattern loss is the difference
between a pattern's best post-exposure episode success and its final success.

## Main result: 8 steps × 12 cycles

Mean ± sample standard deviation across five seeds:

| Method | Final success ↑ | Pseudo-label error ↓ | Self-reinforced error ↓ | Pattern loss ↓ |
|---|---:|---:|---:|---:|
| Frozen | 35.8 ± 5.9 | 46.4 ± 7.1 | 44.9 ± 7.2 | 0.0 ± 0.0 |
| Eager self-distill | 0.0 ± 0.0 | 94.4 ± 1.2 | 94.3 ± 1.2 | 39.6 ± 9.7 |
| Confidence-filtered eager | 54.0 ± 30.2 | 56.8 ± 17.2 | 56.5 ± 17.2 | 26.8 ± 14.4 |
| Mean Teacher | 62.8 ± 12.9 | 49.0 ± 8.1 | 48.2 ± 8.2 | 24.3 ± 9.3 |
| Eager + replay | 56.6 ± 9.3 | 60.4 ± 8.1 | 60.2 ± 8.1 | 10.5 ± 2.6 |
| **Lazy Distillation** | **74.0 ± 17.8** | **43.2 ± 12.5** | **43.0 ± 12.5** | **3.6 ± 3.1** |

Frozen has zero pattern loss mechanically because it never updates; it is an
ability anchor, not a continual-learning solution.

## Scale result: 16 steps × 24 cycles

Mean ± sample standard deviation across three seeds:

| Method | Final success ↑ | Pseudo-label error ↓ | Self-reinforced error ↓ | Pattern loss ↓ |
|---|---:|---:|---:|---:|
| Frozen | 52.9 ± 3.5 | 24.0 ± 3.5 | 21.9 ± 3.5 | 0.0 ± 0.0 |
| Eager self-distill | 0.0 ± 0.0 | 95.3 ± 2.5 | 95.2 ± 2.6 | 19.6 ± 8.6 |
| Confidence-filtered eager | 0.0 ± 0.0 | 93.6 ± 1.7 | 93.5 ± 1.7 | 12.2 ± 3.9 |
| Mean Teacher | 79.1 ± 9.6 | 30.3 ± 9.1 | 29.8 ± 9.1 | 11.3 ± 4.4 |
| Eager + replay | 24.3 ± 4.0 | 69.4 ± 1.6 | 69.3 ± 1.6 | 10.9 ± 7.6 |
| SDAR-style positive-gap gate | 3.3 ± 3.0 | 88.3 ± 4.0 | 88.0 ± 4.1 | 41.0 ± 15.4 |
| **Lazy Distillation** | **79.6 ± 0.8** | **29.9 ± 2.9** | **29.8 ± 2.9** | **6.4 ± 4.9** |

At twice the horizon and twice the number of evolution cycles, Lazy remains
stable across seeds. Mean Teacher is competitive on final success, but Lazy has
lower pattern loss. Eager and confidence-filtered eager both collapse to zero,
showing that confidence filtering alone does not break the recursive feedback
loop.

We also adapt SDAR's detached positive-gap gate to the action-level eager
learner. This is a mechanism baseline, not an exact reproduction of SDAR's
token-level GRPO system. It collapses in the long-horizon stress test: a gate
changes how strongly the model learns a current pseudo-target, but does not
stop each eager update from changing the next wave's collector. Lazy addresses
the missing temporal axis by freezing collection and pooling waves.

## Interpretation

The baselines address only part of the coupled failure:

- Confidence filtering can reject weak targets, but repeated updates can make a
  wrong policy confidently wrong.
- Mean Teacher smooths the teacher, but that teacher still incorporates the
  entire sequential update history.
- Replay improves pattern retention, but it can replay corrupted pseudo-labels.
- Lazy changes the causal timing: it both prevents within-window error feedback
  and makes all wave patterns co-present in the update.

A separate probing run added a confidence gate to Lazy. It hurt the long-horizon
result because the evolved policy could be confidently wrong. We therefore do
not present gating as the core method: the supported merit is delayed frozen
collection plus pooled update.

## Related work and baselines

The implemented paper-derived baselines are confidence-based pseudo-label
selection, Mean Teacher, and experience replay/growing-batch training. STaR,
ReST, ReST-meets-ReAct, and Re-ReST motivate verified or reflected agent
self-training but are discussed rather than falsely presented as exact
reproductions.

The closest concurrent work is *Rethinking Continual Experience Internalization
for Self-Evolving LLM Agents* (Chen et al., 2026), which reports progressive
collapse under multi-iteration experience learning and favors stable
off-policy context distillation. Our narrower preliminary contribution is the
controlled isolation of update timing and the paired, per-cycle diagnosis of
confirmation bias and pattern loss.

## References

1. Radhakrishnan et al. *Enhancing Self-Training Methods*. arXiv:2301.07294, 2023.
2. Rodemann et al. *Approximately Bayes-optimal pseudo-label selection*. UAI, 2023.
3. Tarvainen and Valpola. *Mean Teachers Are Better Role Models*. NeurIPS, 2017.
4. Yoo et al. *Layerwise Proximal Replay*. ICML, 2024.
5. Gulcehre et al. *Reinforced Self-Training (ReST) for Language Modeling*. arXiv:2308.08998, 2023.
6. Aksitov et al. *ReST meets ReAct*. arXiv:2312.10003, 2023.
7. Zelikman et al. *STaR*. arXiv:2203.14465, 2022.
8. Dou et al. *Re-ReST*. arXiv:2406.01495, 2024.
9. Chen et al. *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents*. arXiv:2606.04703, 2026.
10. Shenfeld et al. *Self-Distillation Enables Continual Learning*. arXiv:2601.19897, 2026.
11. Lu et al. *Self-Distilled Agentic Reinforcement Learning*. arXiv:2605.15155, 2026.
12. Wu et al. *SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning*. arXiv:2607.14777, 2026.
13. Yang et al. *OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning*. arXiv:2606.26790, 2026.

## Limitations

This is a procedural stress test with a lightweight policy and simulator oracle.
It provides causal diagnosis, not external validity on a real LLM agent
benchmark. Exact reproductions on ALFWorld, WebShop, or an equivalent long
interactive environment remain future work. The SDAR-style row isolates its
gate only and must not be interpreted as a full SDAR reproduction.
