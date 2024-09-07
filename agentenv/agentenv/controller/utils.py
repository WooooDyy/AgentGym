import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, TypedDict

import numpy as np

from agentenv.controller.agent import Agent
from agentenv.controller.task import BaseTask, ExperienceOutput, GenerationConfig


class ActionFormat(Enum):
    REACT = "react"
    FUNCTION_CALLING = "function_calling"
    CODE_AS_ACTION = "code_as_action"


ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)

INVOKING_FUNCTION_PROMPT = """If you want to invoke a provided function or tool, please reply in the following *JSON* format:
```json
{
    "thought": "I think ...",
    "function_name": "function_name",
    "arguments": <valid json object of args>
}
```
Only reply the *JSON* object, no other text should be present.
"""


def format_function_call_prompt(function_description: Sequence) -> str:
    prompt = "You have the following functions available:\n\n"

    tool_descs = [{"type": "function", "function": f} for f in function_description]
    prompt += "\n".join(
        [json.dumps(f, ensure_ascii=False, indent=2) for f in tool_descs]
    )
    prompt += INVOKING_FUNCTION_PROMPT

    return prompt


@dataclass
class ActionWithTought:
    thought: str
    action: str


@dataclass
class EvaluationOutput:
    experiences: Sequence[ExperienceOutput]
    score: float
    success: float


class BaseAdapter:
    conversation_start_dict: dict[
        ActionFormat, tuple[ConversationMessage, ConversationMessage]
    ]

    @staticmethod
    def parse_react(text: str) -> ActionWithTought:
        """
        ReAct format:
        ```
        Thought:
        I think ...

        Action:
        action
        ```
        """
        invalid_format_flg = False
        _split = text.rsplit("Action:", 1)
        if len(_split) == 0:
            _thought, _action = text
            invalid_format_flg = True
        elif len(_split) == 1:
            if "search[" in text or "click[" in text:
                _thought, _action = "", _split[0]
            else:
                _thought, _action = _split[0], ""
            invalid_format_flg = True
        else:
            assert len(_split) == 2
            _thought, _action = _split

        thought = _thought.split("Thought:")
        if len(thought) == 1:
            thought = thought[0]
            invalid_format_flg = True
        else:
            thought = thought[1].strip()
        action = _action.strip()
        if invalid_format_flg:
            print(
                "The text is not in the correct format. Parsing result may not be accurate."
            )
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        return ActionWithTought(thought, action)

    @staticmethod
    def to_react(action_with_thought: ActionWithTought) -> str:
        return f"Thought:\n{action_with_thought.thought}\n\nAction:\n{action_with_thought.action}"

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        """
        Function Calling format:
        ```json
        {
            "function_name": "function_name",
            "args": {"kwarg1": "value1", "kwarg2": "value2"}
        }
        ```
        """
        raise NotImplementedError

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        """
        Code as Action format:
        ```
        code
        ```
        """
        raise NotImplementedError

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @classmethod
    def action_parser(
        cls, action: str, action_format: ActionFormat
    ) -> str | tuple[str, str]:
        if action_format == ActionFormat.REACT:
            return cls.parse_react(action).action
        elif action_format == ActionFormat.FUNCTION_CALLING:
            return cls.parse_function_calling(action).action
        elif action_format == ActionFormat.CODE_AS_ACTION:
            return cls.parse_code_as_action(action).action
        else:
            raise NotImplementedError


class BaseAgentEnvController:
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
        self.agent = agent
        self.tasks = tasks

    def generate_experience(
        self,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        experience = []
        if isinstance(idxs[0], int):
            experience += self.tasks[0].generate_experience(
                self.agent.model,
                self.agent.tokenizer,
                idxs,
                generation_config,
                max_rounds,
            )
        elif isinstance(idxs[0], Sequence):
            for idx, task in enumerate(self.tasks):
                experience += task.generate_experience(
                    self.agent.model,
                    self.agent.tokenizer,
                    idxs[idx],
                    generation_config,
                    max_rounds,
                )
        else:
            raise ValueError("Incorrect Format for idxs")

        return experience


class Evaluator(BaseAgentEnvController):
    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
    ) -> EvaluationOutput:
        exps = self.generate_experience(
            idxs=(
                idxs
                if idxs is not None
                else [list(range(len(task.clients[0]))) for task in self.tasks]
            ),
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        rewards = np.array([exp.reward for exp in exps])
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=(rewards == 1).mean()
        )


class BaseTrainer(BaseAgentEnvController):
    # def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
    #     super().__init__(agent, tasks)

    def train(self):
        pass

    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] = None,
    ) -> EvaluationOutput:
        exps = self.generate_experience(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        rewards = np.array([exp.reward for exp in exps])
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=(rewards == 1).mean()
        )

    def save_model(self):
        pass
