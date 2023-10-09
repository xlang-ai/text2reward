ENV_ID=$1
EXP_PATH=$2
EXP_NAME=$3

for EXP_SEED in 12345 23451 34512 45123 51234; do
    python sac.py --env_id $ENV_ID \
        --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 500 \
        --train_max_steps 1_000_000 --seed $EXP_SEED --reward_path $EXP_PATH --exp_name $EXP_NAME
done