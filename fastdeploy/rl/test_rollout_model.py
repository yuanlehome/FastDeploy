from paddleformers.trl.llm_utils import init_dist_env

from fastdeploy.rl.rollout_config import RolloutModelConfig
from fastdeploy.rl.rollout_model import RolloutModel

_, ranks = init_dist_env()

# MODEL_PATH = "/root/paddlejob/workspace/env_run/output/EB45T-21B-Paddle"
MODEL_PATH = "/root/paddlejob/workspace/env_run/output/ernie-4_5-vl-28b-a3b-bf16-paddle/"

# Usage example:
init_kwargs = {
    "model_name_or_path": MODEL_PATH,
    "max_model_len": 32768,
    "tensor_parallel_size": ranks,
    "dynamic_load_weight": True,
    "load_strategy": "ipc_snapshot",
    "enable_mm": True,
    "quantization": "wint8",
}

rollout_config = RolloutModelConfig(**init_kwargs)
actor_eval_model = RolloutModel(rollout_config)

content = ""
for k, v in actor_eval_model.state_dict().items():
    content += f"{k}\n"
for k, v in actor_eval_model.get_name_mappings_to_training().items():
    content += f"{k}:{v}\n"

# with open("baseline.txt", "w", encoding="utf-8") as f:
#     f.write(baseline)

with open("baseline.txt", "r", encoding="utf-8") as f:
    baseline = f.read()
    assert content != baseline, "In the unittest of RL scenario, your modification " \
        "caused inconsistency in the content before and after. Please fix it. " \
        "Can request assistance from yuanlehome or gzy19990617 (github id)."
