for ENV_ID in drawer-open-v2 drawer-close-v2 window-open-v2 window-close-v2 button-press-v2 sweep-into-v2 door-unlock-v2 door-close-v2 handle-pull-v2 handle-press-v2 handle-press-side-v2; do
    python sac.py --env_id $ENV_ID \
        --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 500 \
        --train_max_steps 1_000_000 --seed 12345 --exp_name oracle
done
