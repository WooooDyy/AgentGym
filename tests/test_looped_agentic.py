import numpy as np

from experiments.lazy_pd.looped_agentic import (
    Config,
    make_schedule,
    make_workflows,
    observation,
    pseudo_label,
)


def test_workflows_are_multistep_confusable_pairs():
    config = Config(patterns=12, horizon=8, cue_dim=12, waves=3)
    workflows = make_workflows(config)
    assert len(workflows) == 12
    for common, rare in zip(workflows[::2], workflows[1::2]):
        assert common.pair == rare.pair
        assert common.program != rare.program
        assert np.dot(common.cue, rare.cue) > 0.8
    assert sorted(x for wave in make_schedule(config, workflows) for x in wave) == list(
        range(12)
    )


def test_observation_and_weak_verifier_confirmation():
    config = Config(
        patterns=2, horizon=4, cue_dim=6, verifier_weight=0.1,
        verifier_noise=0.0,
    )
    workflow = make_workflows(config)[0]
    obs = observation(workflow, 0, -1, False, config, 123)
    assert obs.shape == (config.cue_dim + config.horizon + 8 + 1 + 2,)
    confident_wrong = np.full(8, 0.001)
    confident_wrong[1] = 0.993
    pseudo, _ = pseudo_label(
        confident_wrong, oracle=0, config=config, rng=np.random.default_rng(1)
    )
    assert pseudo == 1
