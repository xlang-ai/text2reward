ENV_ID=$1

python sac.py --env_id $ENV_ID \
    --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 200 \
    --train_max_steps 8_000_000 --seed 0 --eval_seed 1