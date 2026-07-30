# ALOHA Carrot Simulation Rebuild

## Outcome

Rebuild the carrot simulation to match the available dataset reference frames as
closely as current local evidence allows:

- fixed left ALOHA arm reset, no right arm
- overhead and left wrist camera names aligned with `aloha_sim`
- cup, plate, carrot placement/orientation aligned to `source_ep0/high_01.png`
- stable objects before grasp and after placement
- scripted left arm navigation that grasps the carrot and places it in the cup
- MuJoCo smoke/rollout validation when `dm_control` is available

## Authority

- User request in Vietnamese for full simulation replacement and dataset fidelity.
- Local dataset reference images under
  `outputs/validation/aloha_carrot_sim_rebuild/source_ep0/`.
- Google DeepMind `aloha_sim` reference:
  `https://github.com/google-deepmind/aloha_sim`.
- Local ACT ALOHA assets under `/home/linh/Desktop/octo/act/assets`.

## Current Implementation

- `octo/sim/aloha_carrot_left.py`: deterministic kinematic task state,
  renderer, controller, metrics.
- `octo/sim/aloha_carrot_mujoco.py`: MuJoCo renderer with ACT left-arm meshes,
  no right arm, deterministic wood texture, source-aligned table/mat/object
  geometry, physically mounted wrist camera, continuous warm-start IK, and a
  six-frame source timeline contact sheet.
- `configs/sim/aloha_carrot_left.json`: fixed reset/camera/object contract.
- `docs/sim/aloha_carrot_left.md`: canonical run/validation notes for the new
  left-arm carrot simulation.
- `scripts/run_aloha_carrot_left_sim.py`: scripted rollout and images/GIFs.
- `scripts/audit_aloha_carrot_assets.py`: explicit audit of source frames, ACT
  assets, parquet dataset files, and learned checkpoint files.
- `scripts/validate_aloha_carrot_left_alignment.py`: landmark check against the
  available source image.
- `scripts/validate_aloha_carrot_left_dataset.py`: all-episode parquet contract,
  fixed reset/gripper values, six timeline qpos rows, and exact source-video
  pixel checks.
- `scripts/validate_aloha_carrot_left_mujoco.py`: MuJoCo smoke, kinematic
  rollout, reset bbox/centroid/RGB alignment, wrist timeline, fixed-init FK,
  end-effector tracking, joint smoothness, and wrist-camera rotation checks.
- `scripts/validate_aloha_carrot_left_timeline.py`: six-frame source-aligned
  timeline export/check for `high_01..06` and `wrist_01..06`, plus a contact
  sheet against the available source frames. Includes wrist-view visual probes
  for foreground gripper fingers, mat/backdrop, carrot, plate, and cup by phase.
- `scripts/validate_aloha_carrot_left_visual_similarity.py`: source-vs-sim
  visual feature comparison across the six high/wrist frames using color ratios
  and centroids for backdrop, mat, cup, plate, carrot, and gripper fingers.
- `tests/test_aloha_carrot_left_sim.py`: invariants and optional MuJoCo test.
  Includes a guard that fails if the active runtime reintroduces `from sim_env`,
  `BOX_POSE`, `make_sim_env`, or `aloha-sim-cube-v0`.
- `examples/envs/aloha_sim_env.py`: Gym wrapper now points directly at the new
  carrot task and no longer imports ACT `sim_env`, `BOX_POSE`, or bimanual cube
  tasks. Autonomous episodes now terminate after retreat, not immediately at
  first success.
- `examples/03_eval_finetuned.py` and `examples/03_pt_eval_finetuned.py`: eval
  examples now instantiate `aloha-carrot-left-v0`.

## Remaining Audit

- Default Python unit tests passed:
  `python3 -m unittest tests.test_aloha_carrot_left_sim`
  (`OK`, 14 tests with 3 dependency skips). This includes the asset audit and
  optional dataset-contract fixtures.
- Asset audit passed for current validation assets:
  `python3 scripts/audit_aloha_carrot_assets.py` with 12/12 source high/wrist
  frames present, 15/15 ACT mesh/XML assets present, and 130 parquet episodes.
  It reported `current_validation_ready=true` and
  `full_dataset_policy_audit_ready=false`. Missing full-audit inputs are:
  `checkpoints/octo/full_seed42/49999/weights.pth`,
  `checkpoints/octo/full_seed42/49999/config.json`,
  `checkpoints/octo/full_seed42/49999/dataset_statistics.json`,
  `checkpoints/octo/full_seed42/49999/example_batch.pickle`, and
  `checkpoints/act/full_seed42/policy_step_050000.ckpt`.
- Conda MuJoCo-capable unit tests passed:
  `conda run -n octo_pt python -m unittest tests.test_aloha_carrot_left_sim`
  (`OK`, 14 tests including dataset, Gym env, runtime-removal guard, timeline,
  and MuJoCo rollout checks).
- Example-directory Gym smoke passed:
  `cd examples && conda run -n octo_pt python -c "... gym.make('aloha-carrot-left-v0') ..."`
  with action shape `(4,)`, proprio shape `(7,)`, image shape `(256, 256, 3)`,
  `left_arm_only=True`, `right_arm_present=False`, `done=True`,
  `truncated=False`, 71 steps, and final state `in_cup`.
- Search found no remaining Python runtime references for
  `from sim_env`, `BOX_POSE`, `make_sim_env`, or `aloha-sim-cube-v0`. The file
  name `aloha_sim_env.py` remains as the local wrapper name. Some legacy
  training and latency examples still mention external cube datasets as dataset
  samples, not the active simulation runtime.
- Dataset validation passed:
  `conda run -n octo_pt python scripts/validate_aloha_carrot_left_dataset.py`.
  All 130 episodes and 18,741 frames match metadata at 15 FPS; state/action are
  7D left-arm vectors; fixed reset qpos/action errors are `4.05e-9` and
  `5.71e-8`; six configured source qpos rows have max error `4.82e-9`; and all
  12 source images exactly match episode-0 frames `0,30,60,90,120,148` with
  pixel MAE `0.0`.
- Scripted rollout passed:
  `python3 scripts/run_aloha_carrot_left_sim.py --render --output-dir outputs/simulation/aloha_carrot_left/scripted_smoke`
  with success `true`, max reward `4.0`, final state `in_cup`, 71 steps,
  `left_arm_only=true`, `right_arm_present=false`, and post-place carrot
  translation std `1.11e-16`.
- Alignment validation passed:
  `python3 scripts/validate_aloha_carrot_left_alignment.py`
  with cup/plate/carrot landmark errors all `0.0px` against the available
  source-frame landmarks and visual probes for cup cyan, carrot orange, plate
  green, mat pink, and backdrop green.
- Timeline validation passed:
  `python3 scripts/validate_aloha_carrot_left_timeline.py`
  with all 12 source frames present, expected phases
  `on_plate,on_plate,held,held,in_cup,in_cup`, rollout success, pre-grasp and
  post-place object stability, no carrot spin, wrist visual probes passing, and
  final retreat away from the cup. The generated contact sheet is
  `outputs/validation/aloha_carrot_left_timeline/timeline_contact_sheet.png`.
- Visual similarity validation passed:
  `python3 scripts/validate_aloha_carrot_left_visual_similarity.py` with 49/49
  source-vs-sim feature probes passing across `high_01..06` and
  `wrist_01..06`. The generated contact sheet is
  `outputs/validation/aloha_carrot_left_visual_similarity/visual_similarity_contact_sheet.png`.
- MuJoCo validation passed:
  `conda run -n octo_pt python scripts/validate_aloha_carrot_left_mujoco.py`
  with `qpos_shape=[15]`, `ctrl_shape=[8]`, no right arm, static object delta
  `0.0`, initial FK/config error `4.94e-9`, rollout success `true`, max reward
  `4.0`, final state `in_cup`, and 71 steps. Dynamic bounds passed with max
  end-effector tracking error `0.021038m`, max left-joint step `0.14rad`,
  max wrist-camera rotation `7.51631deg/step`, and final frame-148 reference
  joint error `0.0126rad`.
- MuJoCo object stability passed with pre-grasp carrot translation std
  `6.94e-17`, post-place translation std `2.78e-17`, and carrot quaternion
  delta `0.0`.
- MuJoCo source-aligned visual checks passed for all source/render files, reset
  geometry, reset colors, and wrist timeline framing. The six-frame comparison
  is
  `outputs/validation/aloha_carrot_left_mujoco/mujoco_source_timeline_contact_sheet.png`.
- Generated MuJoCo renders were visually inspected after calibration. The
  overhead view now matches the source table trapezoid, mat/object placement and
  scale; the reset hides proximal meshes outside the dataset camera while
  retaining the distal left-arm chain for approach/grasp; wrist frames use a
  fixed gripper-mounted camera and follow dataset frame-120/frame-148 retreat
  references after placement.

## Known Limits

- The learned Octo and ACT checkpoints are not present in the current worktree,
  so learned-policy fidelity cannot yet be measured.
- The autonomous grasp is a deterministic scripted controller, not a learned
  policy checkpoint rollout.
