# ALOHA carrot Octo finetuning pipeline

This version uses all local episodes for training; no train/val/test split is
created or consumed.

Defaults:

- Source LeRobot dataset: `data/aloha_carrot_easy`
- RLDS dataset: `outputs/derived/aloha_carrot_easy_rlds`
- TFDS split: `train`
- Checkpoints: `checkpoints/octo/aloha_carrot_finetune_seed42`
- Training: `60000` optimizer steps
- Checkpoint interval: `10000` steps
- Window size: `1`
- Action horizon: `20`
- Inputs: `top`, `wrist`, `state`
- Outputs: 7D action

Run the full pipeline:

```bash
conda activate octo_pt
cd /home/linh/Desktop/octo/octo-pytorch
bash scripts/run_aloha_carrot_finetune_pipeline.sh
```

Train 50k instead of 60k:

```bash
NUM_STEPS=50000 SAVE_INTERVAL=10000 bash scripts/run_aloha_carrot_finetune_pipeline.sh
```

Resume to 60k from an existing checkpoint directory:

```bash
RESUME_FROM=checkpoints/octo/aloha_carrot_finetune_seed42 \
NUM_STEPS=60000 \
bash scripts/run_aloha_carrot_finetune_pipeline.sh
```

Main outputs:

- `checkpoints/octo/aloha_carrot_finetune_seed42/10000/weights.pth`
- `checkpoints/octo/aloha_carrot_finetune_seed42/20000/weights.pth`
- `checkpoints/octo/aloha_carrot_finetune_seed42/30000/weights.pth`
- `checkpoints/octo/aloha_carrot_finetune_seed42/40000/weights.pth`
- `checkpoints/octo/aloha_carrot_finetune_seed42/50000/weights.pth`
- `checkpoints/octo/aloha_carrot_finetune_seed42/60000/weights.pth`
- `checkpoints/octo/aloha_carrot_finetune_seed42/metrics/training_metrics.csv`
- `checkpoints/octo/aloha_carrot_finetune_seed42/metrics/loss_curve.png`
- `checkpoints/octo/aloha_carrot_finetune_seed42/metrics/metrics_curves.png`
- `outputs/eval/aloha_carrot_finetune_seed42/checkpoint_metrics.csv`
- `outputs/eval/aloha_carrot_finetune_seed42/checkpoint_eval_curves.png`

Metric notes:

- `train_loss`: masked L1 action loss from the Octo action head.
- `mae_per_dim`: `train_loss / active_action_dims`.
- `mse_per_dim` and `rmse_per_dim`: normalized action MSE/RMSE trends.
- `gripper_accuracy` and `gripper_f1`: checkpoint eval metrics on the final
  action dimension, thresholded at the dataset gripper min/max midpoint.

Replot metrics only:

```bash
python scripts/plot_training_metrics.py \
  --metrics-csv checkpoints/octo/aloha_carrot_finetune_seed42/metrics/training_metrics.csv
```

Evaluate checkpoints only:

```bash
python scripts/evaluate_octo_checkpoint_metrics.py \
  --checkpoint-dir checkpoints/octo/aloha_carrot_finetune_seed42 \
  --data-dir outputs/derived/aloha_carrot_easy_rlds \
  --dataset-name aloha_carrot_easy_rlds \
  --dataset-split train \
  --dataset-statistics outputs/derived/aloha_carrot_easy_rlds/aloha_carrot_easy_rlds/1.0.0/octo_dataset_statistics.json \
  --steps 10000,20000,30000,40000,50000,60000 \
  --device cuda:0
```
