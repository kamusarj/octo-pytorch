# Open-Loop Report: Full-Data and Real Episode-0 Overfit Checkpoints

Re-evaluation date: **2026-08-04**
Status: **complete**

## Protocol

This is teacher-forced open-loop evaluation. At each inference point, a policy
receives recorded RLDS observations, predicts an action chunk, and the evaluator
stitches its first eight actions against ground truth. Predictions are never fed
back to the simulator, so these are imitation metrics rather than success rates.

All reported values are in `original_units`, after unnormalizing with the
checkpoint's own statistics. Blue plot lines are ground truth, orange lines are
predictions, dashed gray lines are state, and purple dots mark inference points.

## Checkpoints and data

| Model | Step | Fine-tuning data | Open-loop data |
|---|---:|---|---|
| `aloha_carrot_w2_h8_stageb_seed42` | 20000 | full data | original RLDS: train ep0, validation ep6 |
| `one_episode_ep0_jitter1cm_command_v4_adapt` | 4500 | real-data episode 0 | episode-0 RLDS: `train` split only |

The step-20000 checkpoint was loaded only for evaluation and was not modified.
The step-4500 checkpoint resumed from step 2500 and continued overfitting the
single real episode-0 demonstration.

## Results

| Checkpoint | Split / episode | MAE | MSE | RMSE | Gripper accuracy |
|---|---|---:|---:|---:|---:|
| step 20000 | train / ep0 | 0.028095 | 0.001925 | 0.043876 | 97.99% |
| step 20000 | validation / ep6 | 0.171922 | 0.097308 | 0.311942 | 96.88% |
| real ep0 overfit step 2500 | train / ep0 | 0.011425 | 0.000353 | 0.018779 | 99.33% |
| **real ep0 overfit step 4500** | **train / ep0** | **0.008951** | **0.000212** | **0.014549** | **99.33%** |

Step 4500 reduces episode-0 MAE by 21.6% from step 2500 and by 68.1% from
step 20000 on that same episode. It is the checkpoint configured in
`scripts/run_octo_two_model_open_loop.sh`.

There is deliberately no step-4500 validation row: the overfit RLDS contains
only a `train` split. Requesting `validation` violates that dataset contract and
TFDS rejects it. A validation plot therefore must not be used to judge whether
the episode-0 overfit worked.

## GT-vs-Prediction Plots

**Step-20000 checkpoint — training episode 0**

![Step-20000 checkpoint, training episode 0](assets/open_loop/step20000_train_gt_vs_pred.png)

**Step-20000 checkpoint — validation episode 6**

![Step-20000 checkpoint, validation episode 6](assets/open_loop/step20000_validation_gt_vs_pred.png)

**Real episode-0 overfit checkpoint at step 4500 — training episode 0**

![Real episode-0 overfit checkpoint at step 4500, training episode 0](assets/open_loop/real_ep0_step4500_train_gt_vs_pred.png)

The first two plots are retained as the step-20000 reference. The third is the
new overfit plot and is the correct one for assessing the episode-0 target; no
step-4500 validation plot exists because that dataset has no validation split.

## Artifacts and reproduction

Use this plot to inspect the actual overfit target:

```text
outputs/eval/open_loop_real_ep0_step4500_train/train/trajectory_000_gt_vs_pred.png
```

It contains 149 episode-0 train steps, 19 inference calls, action horizon 20,
and execution horizon 8. The auditable summary is:

```text
outputs/eval/open_loop_real_ep0_step4500_train/summary.json
```

```bash
/home/linh/anaconda3/envs/octo_pt/bin/python scripts/evaluate_octo_open_loop.py \
  --checkpoint-dir checkpoints/octo/one_episode_ep0_jitter1cm_command_v4_adapt \
  --checkpoint-step 4500 \
  --data-dir outputs/derived/aloha_carrot_ep0_rlds \
  --dataset-name aloha_carrot_easy_rlds \
  --dataset-statistics outputs/derived/aloha_carrot_ep0_rlds/aloha_carrot_easy_rlds/1.0.0/octo_dataset_statistics.json \
  --splits train --traj-ids 0 --steps 200 --execution-horizon 8 --device cuda:0 \
  --output-dir outputs/eval/open_loop_real_ep0_step4500_train
```

`outputs/` is Git-ignored; the summary, CSV, NPZ trace, and plot remain local
for audit.
