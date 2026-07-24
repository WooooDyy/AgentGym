# Reviewer-facing claim and evidence audit

## Mini-outline

1. Task: repeated long-horizon self-evolution from policy-generated experience.
2. Challenge: immediate updates couple wrong pseudo-labels to future data and
   sequentially overwrite workflow patterns.
3. Method: freeze the collector for a multi-wave window, pool all waves, and
   update once.
4. Evidence: oracle diagnostic metrics, timing controls, paper-derived
   baselines, harder 16-step/24-cycle scale study, and per-cycle dynamics.
5. Boundary: controlled procedural benchmark; SDAR-style is a gate adaptation,
   not a full token-level GRPO reproduction.

## Claim–evidence map

- Claim: repeated eager self-distillation amplifies wrong supervision.
  Evidence: eager pseudo-label error is 94.4% (main) and 95.3% (scale), with
  nearly identical self-reinforced error. Status: supported.
- Claim: sequential eager updates erase previously acquired workflows.
  Evidence: eager peak-to-final pattern loss is 39.6% (main); the per-pattern
  metric requires demonstrated earlier success. Status: supported.
- Claim: Lazy addresses both observed failures.
  Evidence: in the scale setting, Lazy has 29.9% pseudo-label error and 6.4%
  pattern loss versus SDAR-style's 88.3% and 41.0%; final success is 79.6%.
  Status: supported in the controlled benchmark.
- Claim: SDAR itself is inferior to Lazy.
  Evidence: not available; only its detached positive-gap gate is adapted.
  Status: unsupported and explicitly not claimed.
- Claim: Lazy generalizes to full LLM agents.
  Evidence: not yet available. Status: needs evidence; stated as future work.

## Five-dimension adversarial review

### Contribution

- Pass: the paper isolates update timing as a causal axis and defines paired
  long-horizon agent diagnostics.
- Risk: delayed/batched training is mechanically simple. The paper therefore
  claims a timing insight and controlled finding, not algorithmic complexity.

### Writing clarity

- Pass: overview figure, metric table, dataset table, exact equations, baseline
  descriptions, and runnable code are included.
- Pass: confirmation bias and pattern loss are distinguished with a concrete
  tool-choice example.

### Experimental strength

- Pass: results cover two horizons, up to 24 cycles, three or five seeds, and
  report standard deviations and failure cases.
- Risk: the environment and policy are procedural/lightweight.

### Evaluation completeness

- Pass: Frozen, Eager, confidence filtering, Mean Teacher, replay, and an
  SDAR-style positive-gap gate are evaluated under matched schedules.
- Needs new experiment: exact SDAR/OPID/SEED comparisons on ALFWorld, WebShop,
  or SearchQA with a language model and RL objective.
- Needs new experiment: window-size ablation and a full Lazy+SDAR combination.

### Method soundness

- Pass: the intervention is low-complexity and does not use oracle labels.
- Risk: delayed updates can slow adaptation under stationary or rapidly
  changing streams; the current study targets non-IID long-horizon evolution.
