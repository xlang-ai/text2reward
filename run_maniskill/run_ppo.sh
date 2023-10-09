ENV_ID=$1

python ppo.py --env_id $ENV_ID \
    --train_num 8 --eval_num 5 --eval_freq 12800 --max_episode_steps 100 \
    --rollout_steps 3200 --train_max_steps 4_000_000 --seed 0 --eval_seed 1