# Lift Low-Dim

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_lowdim_bc      policy=mlp     policy.mlp.hidden_dims=[256]     policy.mlp.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_lowdim/lift/ph/low_dim_converted.hdf5"       training.num_offline_steps=10000     eval.eval_freq=1000       logging.wandb_name=robomimic_lowdim_mlp    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_lowdim_bc      policy=transformer     policy.transformer.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_lowdim/lift/ph/low_dim_converted.hdf5"       training.num_offline_steps=10000     training.chunk_size=4      training.n_action_steps=4     eval.eval_freq=1000       logging.wandb_name=robomimic_lowdim_act    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_lowdim_dp      env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_lowdim/lift/ph/low_dim_converted.hdf5"       training.num_offline_steps=10000     training.chunk_size=4      training.n_action_steps=4      offline_algorithm.obs_encoder_hidden_dims=[]     eval.eval_freq=1000       logging.wandb_name=robomimic_lowdim_dp    logging.wandb_entity=tri

# Square Low-Dim

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_lowdim_bc      policy=mlp     policy.mlp.hidden_dims=[256]     policy.mlp.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_lowdim/square/ph/low_dim_converted.hdf5"       training.num_offline_steps=50000     eval.eval_freq=5000       logging.wandb_name=robomimic_lowdim_mlp    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_lowdim_bc      policy=transformer     policy.transformer.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_lowdim/square/ph/low_dim_converted.hdf5"       training.num_offline_steps=50000     training.chunk_size=8      training.n_action_steps=6     eval.eval_freq=5000       logging.wandb_name=robomimic_lowdim_act    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_lowdim_dp      env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_lowdim/square/ph/low_dim_abs_converted.hdf5"       training.num_offline_steps=50000     training.chunk_size=8      training.n_action_steps=8      offline_algorithm.obs_encoder_hidden_dims=[]     eval.eval_freq=5000       logging.wandb_name=robomimic_lowdim_dp    logging.wandb_entity=tri

# Can

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_image_dp      env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_image/can/ph/image_converted.hdf5"      training.chunk_size=10      training.n_action_steps=8       logging.wandb_name=robomimic_image_dp    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_lowdim_dp      env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_lowdim/can/ph/low_dim_converted.hdf5"      training.chunk_size=10      training.n_action_steps=8       logging.wandb_name=robomimic_lowdim_dp    logging.wandb_entity=tri

# Lift Image

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_image_bc      policy=mlp     policy.mlp.hidden_dims=[256]     policy.mlp.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_image/lift/ph/image_converted.hdf5"       training.num_offline_steps=50000     eval.eval_freq=5000       logging.wandb_name=robomimic_image_mlp    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_image_bc      policy=transformer     policy.transformer.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_image/lift/ph/image_converted.hdf5"       training.num_offline_steps=50000     training.chunk_size=4      training.n_action_steps=4     eval.eval_freq=5000       logging.wandb_name=robomimic_image_act    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_image_dp      env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_image/lift/ph/image_abs_converted.hdf5"       training.num_offline_steps=50000     training.chunk_size=4      training.n_action_steps=4      offline_algorithm.obs_encoder_hidden_dims=[]     eval.eval_freq=5000       logging.wandb_name=robomimic_image_dp    logging.wandb_entity=tri

# Square Image

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_image_bc      policy=mlp     policy.mlp.hidden_dims=[256]     policy.mlp.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_image/square/ph/image_converted.hdf5"       training.num_offline_steps=100000     eval.eval_freq=10000       logging.wandb_name=robomimic_image_mlp    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_image_bc      policy=transformer     policy.transformer.dropout_rate=0.1     env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_image/square/ph/image_converted.hdf5"       training.num_offline_steps=100000     training.chunk_size=8      training.n_action_steps=6     eval.eval_freq=10000       logging.wandb_name=robomimic_image_act    logging.wandb_entity=tri

uv run python scripts/train_supervised.py     --config-path=../robometer_policy_learning/configs    --config-name=robomimic_image_dp      env.h5_dataset_path="/home/yigitkorkmaz/datasets/robomimic_image/square/ph/image_abs_converted.hdf5"       training.num_offline_steps=100000     training.chunk_size=8      training.n_action_steps=6      offline_algorithm.obs_encoder_hidden_dims=[]     eval.eval_freq=10000       logging.wandb_name=robomimic_image_dp    logging.wandb_entity=tri

## HitL

# Diffusion MILE

uv run python scripts/train_hitl.py     --config-name robomimic_hitl \
    load_dir=/home/yigitkorkmaz/robometer-policy-learning/outputs/2026-06-11/16-17-39 \
    checkpoint=20000 \
    offline_algorithm.intervention_cost=1.5 \
    offline_algorithm.actor_optimizer_lr=1.0e-4 \
    offline_algorithm.batch_size=256 \
    hitl.precollected_hitl_dataset=data/square_hitl_rollouts_30_demos.hdf5 \
    hitl.num_iterations=1 \
    hitl.rollouts_per_iter=0 \
    hitl.train_steps_per_iter=1000 \
    hitl.save_interval=1 \
    eval.eval_freq=100 \
    logging.wandb_name=square_human_diffusion_mile

# HG-DAgger

uv run python scripts/train_hitl.py     --config-name robomimic_hitl \
    load_dir=/home/yigitkorkmaz/robometer-policy-learning/outputs/2026-06-11/16-17-39 \
    checkpoint=20000 \
    algorithm@offline_algorithm=dp \
    alg.offline_alg_name=dp \
    offline_algorithm.actor_optimizer_lr=1.0e-4 \
    offline_algorithm.batch_size=256 \
    hitl.precollected_hitl_dataset=data/square_hitl_rollouts_30_demos.hdf5 \
    hitl.num_iterations=1 \
    hitl.rollouts_per_iter=0 \
    hitl.train_steps_per_iter=1000 \
    hitl.save_interval=1 \
    eval.eval_freq=100 \
    logging.wandb_name=square_human_hgdagger



# Online Human Flow HG-DAgger

uv run python scripts/train_hitl.py     --config-name robomimic_hitl     \
    load_dir=/home/yigitkorkmaz/robometer-policy-learning/checkpoints/square_low_dim/2026-06-22_15-58-08     \
    checkpoint=20000    \
    algorithm@offline_algorithm=flow     \
    alg.offline_alg_name=flow     \
    offline_algorithm.actor_optimizer_lr=1.0e-5     \
    offline_algorithm.batch_size=256           \
    hitl.num_iterations=5     \
    hitl.rollouts_per_iter=10     \
    hitl.train_steps_per_iter=1000     \
    hitl.save_interval=1    \
    hitl.store_only_human=true     \
    hitl.offline_mode=warmup     \
    hitl.warmup_rollouts=10     \
    eval.eval_freq=5   \
    logging.wandb_name=square_online_human_flow_hgdagger


# Offline Human Flow MILE

uv run python scripts/train_hitl.py     --config-name robomimic_hitl     \
    algorithm@offline_algorithm=flow_mile     \
    alg.offline_alg_name=flow_mile     \
    load_dir=/home/yigitkorkmaz/robometer-policy-learning/checkpoints/square_low_dim/2026-06-22_15-58-08     \
    checkpoint=10000      \
    offline_algorithm.condition_intervention_on_action=true      \
    offline_algorithm.anchor_loss_weight=0.1     \
    offline_algorithm.lambda_intervention=1     \
    offline_algorithm.intervention_cost=1.0     \
    offline_algorithm.actor_optimizer_lr=1.0e-5     \
    offline_algorithm.batch_size=256     \
    offline_algorithm.log_sample_metrics_every=200      \
    hitl.precollected_hitl_dataset=data/real_square_hitl_rollouts_flow_30_demos.hdf5     \
    hitl.segment_by_intervention=false     \
    hitl.num_iterations=1     \
    hitl.rollouts_per_iter=0     \
    hitl.train_steps_per_iter=1000     \
    hitl.save_interval=1     \
    eval.eval_freq=200     \
    logging.wandb_name=square_human_flow_mile

# Online Human Flow MILE

uv run python scripts/train_hitl.py     --config-name robomimic_hitl     \
    algorithm@offline_algorithm=flow_mile     \
    alg.offline_alg_name=flow_mile     \
    load_dir=/home/yigitkorkmaz/robometer-policy-learning/checkpoints/square_low_dim/2026-06-22_15-58-08     \
    checkpoint=20000      \
    offline_algorithm.condition_intervention_on_action=true      \
    offline_algorithm.anchor_loss_weight=0.1     \
    offline_algorithm.lambda_intervention=1     \
    offline_algorithm.intervention_cost=1.0     \
    offline_algorithm.actor_optimizer_lr=1.0e-5     \
    offline_algorithm.batch_size=256     \
    hitl.segment_by_intervention=false     \
    hitl.num_iterations=5     \
    hitl.rollouts_per_iter=10     \
    hitl.train_steps_per_iter=1000     \
    hitl.save_interval=1     \
    hitl.store_only_human=false     \
    hitl.keep_only_hitl_rollouts=true     \
    eval.eval_freq=5     \
    logging.wandb_name=square_online_human_flow_mile