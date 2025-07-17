export PYTHONPATH=/root/paddlejob/workspace/env_run/output/liuyuanle/FastDeploy:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=6,7

python -m paddle.distributed.launch --gpus ${CUDA_VISIBLE_DEVICES} test_rollout_model.py
