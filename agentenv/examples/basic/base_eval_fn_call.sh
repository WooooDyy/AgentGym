# Evaluation args

model_path=""
inference_file="AgentEval/webshop_test.json"
output_file="webshop_test_result_test.jsonl"
task_name="webshop"
seed="42"

# environment parameters
max_round="20"
env_server_base="http://127.0.0.1:8000"

python -u base_eval_fn_call.py \
        --model_path "${model_path}" \
        --inference_file "${inference_file}" \
        --output_file "${output_file}" \
        --task_name "${task_name}" \
        --seed "${seed}" \
        --max_round "${max_round}" \
        --env_server_base "${env_server_base}"
