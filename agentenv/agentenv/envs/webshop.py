import json
from typing import Any, Mapping

import requests

from agentenv.controller import (
    ActionFormat,
    ActionWithTought,
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    ConversationMessage,
    StepOutput,
    format_function_call_prompt,
)

WEBSHOP_FUNCTION_DESCRIPTION = [
    {
        "name": "search",
        "description": "If the search bar is on the page, you can use this function to search for a product. If the action is not valid, perform nothing.",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Keywords in search are up to you. Remember that your keywords in search should be carefully designed.",
                }
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "click",
        "description": "Click on a button.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "The item to click. The item should be one of the cilickable values on the page.",
                }
            },
            "required": ["item"],
        },
    },
]


class WebshopAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought:\nI think ... \n\nAction: \nclick[something]",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f"You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be done by invoking a function.\n\n{format_function_call_prompt(WEBSHOP_FUNCTION_DESCRIPTION)}\n\n\nIf the page remains unchanged, it might indicate that your action is invalid.",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "TODO: Add instructions for code as action",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        _fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
        )
        thought = _fn_call["thought"]
        fn_name = _fn_call["function_name"]
        args = _fn_call["arguments"]
        if fn_name not in ["search", "click"]:
            raise ValueError("Invalid function name.")
        if fn_name == "search":
            action = f"search[{args['keywords']}]"
        else:
            action = f"click[{args['item']}]"
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        if action_with_thought.action.startswith("search"):
            fn_name = "search"
            args = {"keywords": action_with_thought.action.split("[")[-1].split("]")[0]}
        elif action_with_thought.action.startswith("click"):
            fn_name = "click"
            args = {"item": action_with_thought.action.split("[")[-1].split("]")[0]}
        else:
            raise ValueError("Invalid action.")
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": fn_name,
                "arguments": args,
            },
            ensure_ascii=False,
            indent=2,
        )

    # @staticmethod
    # def parse_code_as_action(text: str) -> ActionWithTought:
    #     pass

    # @staticmethod
    # def to_code_as_action(action_with_thought: ActionWithTought) -> str:
    #     pass


class WebshopEnvClient(BaseEnvClient):
    adapter_cls = WebshopAdapter

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {ok}")
        self.conversation_start = self.adapter_cls.conversation_start_dict[
            self.action_format
        ]
        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        max_retries = 5
        for attempt in range(max_retries):
            res = requests.post(
                f"{self.env_server_base}/{path}",
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 503:
                import time

                time.sleep(0.1)
            elif res.status_code == 200:
                break
            else:
                print("---------------------")
                print(res.status_code)
                print(data)
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        try:
            action = WebshopAdapter.action_parser(action, self.action_format)
        except Exception as e:
            print(e, action)
            return StepOutput(
                state="Invalid Action.\n\n" + self.observe(), reward=0.0, done=False
            )
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["state"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int) -> dict[str, Any]:
        response = self._post("reset", {"session_id": idx})
        response[0] = self.observe()
        return response


class WebshopTask(BaseTask):
    env_client_cls = WebshopEnvClient
    env_name = "WebShop"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
