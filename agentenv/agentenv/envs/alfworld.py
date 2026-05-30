import json
from typing import Any, Mapping
import re

import requests
from requests.exceptions import RequestException

from agentenv.controller import (
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    extract_python_code_blocks,
    format_code_as_action_prompt,
    format_function_call_prompt,
    parse_python_code_comments,
)
from agentenv.controller.types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    StepOutput,
)

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


ALFWORLD_FUNCTION_DESCRIPTION = [
        {
        "name":"goto",
        "description":"Move towards a specific receptacle or to a specific location.",
        "parameters":{
            "type":"object",
            "properties":{
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to move towards or the location you want to arrive at.",
                },  
            },
            "required":["recep"],
        }
    },
    {
        "name":"take",
        "description":"Picks up an object from a specified receptacle.",
        "parameters":{
            "type":"object",
            "properties":{
                "obj":{
                    "type":"string",
                    "description":"The object you want to pick up.",
                },
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to pick up object from.",
                }
            },
            "required":["obj","recep"],
        }
    },
    {
        "name":"put",
        "description":"Puts an object on a specified receptacle.",
        "parameters":{
            "type":"object",
            "properties":{
                "obj":{
                    "type":"string",
                    "description":"The object you want to place.",
                },
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to put object on.",
                }
            },
            "required":["obj","recep"],
        }
    },
    {
        "name":"open",
        "description":"Opens a receptacle to reveal its contents.",
        "parameters":{
            "type":"object",
            "properties":{
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to open.",
                }
            },
            "required":["obj"]
        }
    },
    {
        "name":"close",
        "description":"Closes a receptacle.",
        "parameters":{
            "type":"object",
            "properties":{
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to close.",
                }
            },
            "required":["recep"]
        }
    },
    {
        "name": "look",
        "description": "Describe the current situation. Provide information such as what you are facing and what are things next to it.",
        "parameters":{
            "type": "object",
            "properties":{},
            "required": ["obj"]
        },
    },
    {
        "name":"toggle",
        "description":"Switches an object on or off.",
        "parameters":{
            "type":"object",
            "properties":{
                "obj":{
                    "type":"string",
                    "description":"The object you want to toggle.",
                },
            },
            "required":["obj"]
        }
    },
    {
        "name":"heat",
        "description":"Heats an object using a specified receptacle.",
        "parameter":{
            "type":"object",
            "properties":{
                "obj":{
                    "type":"string",
                    "description":"The object you want to heat.",
                },
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to use to heat object.",
                }
            },
            "required":["obj","recep"]
        }
    },
    {
        "name":"cool",
        "description":"Cools an object using a specified receptacle.",
        "parameter":{
            "type":"object",
            "properties":{
                "obj":{
                    "type":"string",
                    "description":"The object you want to cool.",
                },
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to use to cool object.",
                }
            },
            "required":["obj","recep"]
        }
    },
    {
        "name":"clean",
        "description":"Cleans an object using a specified receptacle.",
        "parameter":{
            "type":"object",
            "properties":{
                "obj":{
                    "type":"string",
                    "description":"The object you want to clean.",
                },
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to use to clean object.",
                }
            },
            "required":["obj","recep"]
        }
    },
    {
        "name":"inventory",
        "description":"Displays the list of objects currently being carried by you.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name":"examine",
        "description":"Provides a description of the objects present on or in a receptacle.",
        "parameters":{
            "type":"object",
            "properties":{
                "recep":{
                    "type":"string",
                    "description":"The receptacle you want to get more information.",
                },  
                "obj":{
                    "type":"string",
                    "description":"The object(like a desklamp in order to look clearly) you want use to examine a receptacle.",
                },  
            },
            "required":["recep"],
        }
    },
    {
        "name":"use",
        "description":"Uses an object as a tool to accomplish a certain goal.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type":"string",
                    "description":"The object(like a desklamp in order to look clearly) you want use.",
                },  
            },
            "required":["obj"],
        }
    }
]

class AlfWorldAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT:(
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
                }
            ),
            ConversationMessage(
                {
                    "from": "gpt",
                    "loss": False,
                    "value": "OK. I'll follow your instructions and try my best to solve the task.",
                }
            ),
        ),
        ActionFormat.FUNCTION_CALLING:(
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. Note that you should not choose actions and objects/receptacles not listed in the first turn. An action should be done by invoking an function.\n\n {format_function_call_prompt(ALFWORLD_FUNCTION_DESCRIPTION)}\n\n\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
                }
            ),
            ConversationMessage(
                {
                    "from": "gpt",
                    "loss": False,
                    "value": "OK. I'll follow your instructions and try my best to solve the task.",
                }
            ),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. Note that you should not choose actions and objects/receptacles not listed in the first turn. You can perform one of these actions by writing python code to invoke a function.\n\n {format_function_call_prompt(ALFWORLD_FUNCTION_DESCRIPTION)}\n\n\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }

    valid_functions_args = {
        'goto': ["recep"], 
        'take': ["obj", "recep"], 
        'put': ["obj", "recep"], 
        'toggle': ["obj"], 
        'open': ["recep"], 
        'close': ["recep"], 
        'heat': ["obj", "recep"], 
        'cool': ["obj", "recep"], 
        'clean': ["obj", "recep"], 
        'examine': ["recep", "obj"], 
        'inventory': [], 
        "look": [],
        'use':["obj"]
    }

    function_to_name = {
        'goto': 'go to', 
        'take': 'take', 
        'put': 'put', 
        'toggle': 'toggle', 
        'open': 'open', 
        'close': 'close', 
        'heat': 'heat', 
        'cool': 'cool', 
        'clean': 'clean', 
        'examine': 'examine', 
        "look": "look",
        'inventory': 'inventory', 
        'use':'use'
    }

    conjunction_words = {
        "take": "from",
        "put": "in/on",
        "heat": "with",
        "cool": "with",
        "clean": "with",
        "examine": "with"
    }

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        _fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
        )
        thought = _fn_call["thought"]
        fn_name = _fn_call["function_name"].strip()
        args = _fn_call["arguments"]

        if fn_name not in AlfWorldAdapter.valid_functions_args:
            raise ValueError("Invalid function name.")
        arg_ls = AlfWorldAdapter.valid_functions_args[fn_name]
        if len(args) > len(arg_ls):
            raise TypeError(f"Got unexpected arguments. Function {fn_name} has {len(arg_ls)} argument(s) but got {len(args)}.")
        if len(args) == 1:
            # open door
            action_name = AlfWorldAdapter.function_to_name[fn_name]
            arg = args[arg_ls[0]]
            action = f'{action_name} {arg}'
        elif len(args) == 0:
            # inventory
            action = f'{AlfWorldAdapter.function_to_name[fn_name]}'
        else:  # two arguments
            # take mug from desk
            action_name = AlfWorldAdapter.function_to_name[fn_name]
            conjunction = AlfWorldAdapter.conjunction_words[fn_name]
            action = f'{action_name} {args[arg_ls[0]]} {conjunction} {args[arg_ls[1]]}'
        return ActionWithTought(thought=thought, action=action)
    
    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        valid_action_flag = False
        fn_name = ''
        action_name = ''
        for k, v in AlfWorldAdapter.function_to_name.items():
            if action_with_thought.action.startswith(v):
                valid_action_flag = True
                fn_name = k
                action_name = v
                break
        if not valid_action_flag:
            raise ValueError(f"{action_with_thought.action}: Invalid action.")
        # inventory
        # open door to kitchen / toggle switch wall
        # heat mug with microwave 
        arg_ls = AlfWorldAdapter.valid_functions_args[fn_name]
        str_arg = action_with_thought.action.replace(action_name, '', 1).strip()
        if fn_name in AlfWorldAdapter.conjunction_words:
            separator = AlfWorldAdapter.conjunction_words[fn_name]
            str_arg_ls = re.split(fr'\s+{separator}\s+', str_arg)
            str_arg_ls = [s.strip() for s in str_arg_ls]
        else:
            str_arg_ls = [str_arg.strip()] if len(str_arg) else []
        
        if len(str_arg_ls) > len(arg_ls):
            raise TypeError(f"Got unexpected arguments. function {fn_name} expected {len(arg_ls)} but got {len(str_arg_ls)}.")

        if len(str_arg_ls) == 0:
            args = {}
        elif len(str_arg_ls) == 1:
            args = {
                arg_ls[0]: str_arg_ls[0]
            }
        else:
            args = {
                arg_ls[0]: str_arg_ls[0],
                arg_ls[1]: str_arg_ls[1]
            }
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": fn_name,
                "arguments": args
            },
            ensure_ascii=False,
            indent=2,
        )
    
    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        def goto(recep: str):
            action_name = AlfWorldAdapter.function_to_name["goto"]
            return f"{action_name} {recep}"
        
        def take(obj: str, recep: str):
            action_name = AlfWorldAdapter.function_to_name["take"]
            conjunction = AlfWorldAdapter.conjunction_words["take"]
            return f"{action_name} {obj} {conjunction} {recep}"
        
        def put(obj: str, recep: str):
            action_name = AlfWorldAdapter.function_to_name["put"]
            conjunction = AlfWorldAdapter.conjunction_words["put"]
            return f"{action_name} {obj} {conjunction} {recep}"
        
        def toggle(recep: str):
            action_name = AlfWorldAdapter.function_to_name["toggle"]
            return f"{action_name} {recep}"
        
        def open(recep: str):
            action_name = AlfWorldAdapter.function_to_name["open"]
            return f"{action_name} {recep}"
        
        def close(recep: str):
            action_name = AlfWorldAdapter.function_to_name["close"]
            return f"{action_name} {recep}"
        
        def heat(obj: str, recep: str):
            action_name = AlfWorldAdapter.function_to_name["heat"]
            conjunction = AlfWorldAdapter.conjunction_words["heat"]
            return f"{action_name} {obj} {conjunction} {recep}"
        
        def cool(obj: str, recep: str):
            action_name = AlfWorldAdapter.function_to_name["cool"]
            conjunction = AlfWorldAdapter.conjunction_words["cool"]
            return f"{action_name} {obj} {conjunction} {recep}"
        
        def clean(obj: str, recep: str):
            action_name = AlfWorldAdapter.function_to_name["clean"]
            conjunction = AlfWorldAdapter.conjunction_words["clean"]
            return f"{action_name} {obj} {conjunction} {recep}"
        
        def examine(recep: str, obj: str=''): # obj is optional
            action_name = AlfWorldAdapter.function_to_name["examine"]
            conjunction = AlfWorldAdapter.conjunction_words["examine"]
            return f"{action_name} {recep} {conjunction} {obj}" if obj else f"{action_name} {recep}"
        
        def inventory():
            action_name = AlfWorldAdapter.function_to_name["inventory"]
            return f"{action_name}"
        
        def look():
            action_name = AlfWorldAdapter.function_to_name["look"]
            return f"{action_name}"
        
        def use(obj:str):
            action_name = AlfWorldAdapter.function_to_name["use"]
            return f"{action_name} {obj}"
        code = extract_python_code_blocks(text)

        try:
            action = eval(code, {
                "goto": goto,
                "take": take,
                "put": put,
                "toggle": toggle,
                "open": open,
                "close": close,
                "heat": heat,
                "cool": cool,
                "clean": clean,
                "examine": examine,
                "inventory": inventory,
                "look": look,
                "use": use
            })
        except Exception as e:
            print(e)
            raise ValueError(f"Invalid action:{code}")
        thought = parse_python_code_comments(code)
        return ActionWithTought(thought=thought, action=action)


    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        text = f"```python\n#{action_with_thought.thought}\n"
        valid_action_flag = False
        fn_name = ''
        action_name = ''
        for k, v in AlfWorldAdapter.function_to_name.items():
            if action_with_thought.action.startswith(v):
                valid_action_flag = True
                fn_name = k
                action_name = v
                break
        if not valid_action_flag:
            raise ValueError(f"{action_with_thought.action}: Invalid action.")
        # inventory
        # open door to kitchen / toggle switch wall
        # heat mug with microwave 
        arg_ls = AlfWorldAdapter.valid_functions_args[fn_name]
        str_arg = action_with_thought.action.replace(action_name, '', 1).strip()
        if fn_name in AlfWorldAdapter.conjunction_words:
            separator = AlfWorldAdapter.conjunction_words[fn_name]
            str_arg_ls = re.split(fr'\s+{separator}\s+', str_arg)
            str_arg_ls = [s.strip() for s in str_arg_ls]
        else:
            str_arg_ls = [str_arg.strip()] if len(str_arg) else []
        
        if len(str_arg_ls) > len(arg_ls):
            raise TypeError(f"Got unexpected arguments. function {fn_name} expected {len(arg_ls)} but got {len(str_arg_ls)}.")

        if len(str_arg_ls) == 0:
            text += f"{fn_name}()"
        elif len(str_arg_ls) == 1:
            text += f"{fn_name}({repr(f'{str_arg_ls[0]}')})"
        else:
            text += f"{fn_name}({repr(f'{str_arg_ls[0]}')},{repr(f'{str_arg_ls[1]}')})"
        text += "\n```"
        return text
        


class AlfWorldEnvClient(BaseEnvClient):
    adapter_cls = AlfWorldAdapter
    
    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {ok}")
        
        self.conversation_start = self.adapter_cls.conversation_start_dict[
            self.action_format
        ]
        
        ok = ok.json()
        # print(ok)
        self.env_id = ok["id"]
        self.info = None

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return f"{self.info['observation']}\nAVAILABLE ACTIONS: {','.join(self.info['available_actions'])}"

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        try:
            action = self.adapter_cls.action_parser(action, self.action_format)
        except Exception as e:
            print(e, action)
            return StepOutput(
                state="Invalid Action.\n\n" + self.observe(), reward=0.0, done=False
            )
        # print(f"Action: {action}")
        response = self._post("step", {"action": action})
        # print(response)
        if "observation" not in response:
            raise KeyError(
                f"AlfWorld step response missing 'observation': {json.dumps(response, ensure_ascii=True)}"
            )
        self.info = {
            "observation": response["observation"],
            "available_actions": response.get("available_actions", []),
            "reward": response.get("reward", 0.0),
            "done": response.get("done", False),
        }
        return StepOutput(
            state=response["observation"],
            reward=response.get("reward", 0.0),
            done=response.get("done", False),
        )

    def reset(self, game: int, world_type: str = "Text") -> dict[str, Any]:
        response = {}
        for retry in range(3):
            response = self._post("reset", {"game": game, "world_type": world_type})
            if "observation" in response:
                break
            print(
                f"[AlfWorldEnvClient.reset] malformed response (attempt {retry + 1}/3): "
                f"{json.dumps(response, ensure_ascii=True)}"
            )
        if "observation" not in response:
            raise KeyError(
                "AlfWorld reset failed after 3 attempts; "
                f"missing 'observation' in response: {json.dumps(response, ensure_ascii=True)}"
            )
        self.info = {
            "observation": response["observation"],
            "available_actions": response.get("available_actions", []),
            "reward": 0,
            "done": False,
        }
        return response

    def close(self):
        response = self._post("close",{})
        return response

class AlfWorldTask(BaseTask):
    env_client_cls = AlfWorldEnvClient
    env_name = "AlfWorld"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
