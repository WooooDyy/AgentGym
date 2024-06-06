# Self Evolution

AgentGYM proposes the AgentEvol algorithm, enabling the model to evolve itself.

## Launch the EnvServer

Self-evolution requires the model to explore in the environmens. Please refer to [01-evaluation](docs/tutorials/01-evaluation.md) and `README.md` in each environment server folder to start the corresponding environment server. If you launches multiple servers, please note the port numbers.

## Self Evolution

`agentenv/examples/self_improve` gives a reference implementation of the self-evolution algorithm. Please refer to `examples/self_improve/train_self_improve_multi_task.sh` to start training.
