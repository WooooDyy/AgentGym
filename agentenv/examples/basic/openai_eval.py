import json
import time
from dataclasses import dataclass, field

import jsonlines
import transformers
from tqdm import tqdm

from agentenv.controller import (
    APIAgent,
    Evaluator,
)
from agentenv.envs import (
    AcademiaTask,
    AlfWorldTask,
    BabyAITask,
    MazeTask,
    MovieTask,
    SearchQATask,
    SciworldTask,
    SheetTask,
    SqlGymTask,
    TextCraftTask,
    TodoTask,
    WeatherTask,
    WebarenaTask,
    WebshopTask,
    WordleTask,
)

import os
import random


def is_success_reward(reward: float) -> int:
    return int(reward in (1, 100))


@dataclass
class EvalArguments:
    api_key: str
    base_url: str
    model: str
    inference_file: str = field(metadata={"help": "Test dataset."})
    output_dir: str
    max_tokens: int = field(default=4096)
    temperature: float = field(default=1)
    top_p: float = field(default=1)
    task_name: str = field(
        default="webshop", metadata={"help": "Task name for evaluation"}
    )

    # conversation rounds
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    # environment parameters
    env_server_base: str = field(default=None)
    data_len: int = field(default=200)
    timeout: int = field(default=2400)


def main(args):

    DATA_PATH = args["inference_file"]

    # task_name - task dict
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "babyai": BabyAITask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
        "sqlgym": SqlGymTask,
        "maze": MazeTask,
        "wordle": WordleTask,
        "weather": WeatherTask,
        "todo": TodoTask,
        "movie": MovieTask,
        "sheet": SheetTask,
        "academia": AcademiaTask,
        "searchqa": SearchQATask,
    }

    # select task according to the name
    task_class = task_classes.get(args["task_name"].lower(), None)
    if task_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")

    # set environment parameters
    env_args = {
        "env_server_base": args["env_server_base"],
        "data_len": args["data_len"],
        "timeout": args["timeout"],
    }

    # set env client
    evaluator = Evaluator(
        APIAgent(
            api_key=args["api_key"],
            base_url=args["base_url"],
            model=args["model"],
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
            top_p=args["top_p"],
        ),
        [task_class(client_args=env_args, n_clients=1)],
    )

    with open(DATA_PATH, "r") as file:
        test_data = json.load(file)

    data_idxs = [int(item["item_id"].split("_")[-1]) for item in test_data]
    random.shuffle(data_idxs)

    total_score = 0.0
    total_success = 0.0
    start_time = time.time()
    os.makedirs(args["output_dir"], exist_ok=True)
    for data_idx in tqdm(data_idxs, total=len(data_idxs), desc="[Evaluation Loop]"):
        try:
            with open(os.path.join(args["output_dir"], f"{args['task_name']}_{data_idx}.json"), 'r') as f:
                item = json.load(f)
                total_score += item["reward"]
                total_success += is_success_reward(item["reward"])
            continue
        except:
            pass

        while True:
            try:
                exps = evaluator.eval(
                    max_rounds=args["max_round"],
                    idxs=[data_idx],
                )
                break
            except Exception as e:
                print(e)
                continue
        total_score += exps.score
        total_success += exps.success

        cur_experiences = exps.experiences
        # write inference results to file
        with open(os.path.join(args["output_dir"], f"{args['task_name']}_{data_idx}.json"), 'w') as f:
            for exp in cur_experiences:
                conversation = exp.conversation
                cur_reward = exp.reward
                cur_success = is_success_reward(cur_reward)
                item_id = f"{args['task_name']}_{data_idx}"
                json.dump({
                    "conversations": conversation,
                    "item_id": item_id,
                    "reward": cur_reward,
                    "success": cur_success,
                }, f, ensure_ascii=False, indent=4)
    process_time = time.time() - start_time

    Score = total_score / len(data_idxs)
    Success = total_success / len(data_idxs)
    print("\n\n==== EVALUATION ====\n")
    print(f"Score: {Score}")
    print(f"Success: {Success}")
    print(f"Time: {process_time} seconds")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = vars(args)
    print(args)
    print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
