# Lazy Policy Distillation prototype

This experiment implements the method described in the research proposal as an
actual frozen-language-model policy with trainable low-rank action adapters.
It is deliberately a controlled multi-generation instrument environment: that
makes the competence/discoverability trade-off measurable without conflating it
with failures of a weak model to solve ScienceWorld.

Implemented mechanisms:

- student-sampled on-policy trajectories;
- frozen self-teacher scoring the same action under local hindsight;
- per-action Monte-Carlo reverse-KL with clipped importance correction;
- verifier, signal, consistency, sample-count, and independent-task gates;
- version-bounded buffers that are cleared after each adapter update;
- local family-specific LoRA action adapters over a frozen LM;
- episode-level frozen-base/local-adapter exploration mixture;
- current `Success@1`, future `Discovery@B`, and action entropy.

Run on a GPU:

```bash
conda activate AgentGym
python -m experiments.lazy_pd.lazy_pd \
  --methods frozen eager_global delayed_global immediate_local \
            lazy_local_no_base lazy_local
```

The JSON artifact records the exact config, GPU, per-generation gates, adapter
versions, update losses, and metrics.
