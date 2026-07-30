# ALOHA Carrot Left Simulation

This is the canonical simulation entry point for the dataset-aligned carrot task.
It replaces the old external ACT cube runtime in this repository.

## Contract

- Robot: ALOHA left arm only.
- Removed runtime dependency: external ACT `sim_env`, `BOX_POSE`, and
  `make_sim_env`.
- Gym ID: `aloha-carrot-left-v0`.
- Cameras: `overhead_cam`, `wrist_cam_left`.
- Observation images: `image_primary`, `image_wrist`.
- Proprio: 7 dataset values (`6 joint + 1 gripper`). MuJoCo's two finger slide
  joints remain internal to the renderer.
- Manual action: `[target_x, target_y, target_z, follower_gripper]`.
- Default behavior: autonomous scripted controller.
- Reset: fixed object positions and fixed initial arm pose. The configured
  end-effector reset is the MuJoCo FK result of the fixed left-arm qpos.
- Objects: kinematic carrot/cup/plate state with no pre-grasp or post-place
  jitter, roll, or spin.
- Retreat: follows the episode-0 frame-120/frame-148 end-effector references
  after placing the carrot.

## Run

```bash
python3 scripts/run_aloha_carrot_left_sim.py --render \
  --output-dir outputs/simulation/aloha_carrot_left/scripted_smoke
```

Generated artifacts:

- `primary_reset.png`, `primary_final.png`
- `wrist_reset.png`, `wrist_final.png`
- `primary_rollout.gif`, `wrist_rollout.gif`
- `metrics.json`

## Validate

Fast validation:

```bash
python3 -m unittest tests.test_aloha_carrot_left_sim
python3 scripts/audit_aloha_carrot_assets.py
conda run -n octo_pt python scripts/validate_aloha_carrot_left_dataset.py
python3 scripts/validate_aloha_carrot_left_alignment.py
python3 scripts/validate_aloha_carrot_left_timeline.py
python3 scripts/validate_aloha_carrot_left_visual_similarity.py
```

The asset audit writes:

```text
outputs/validation/aloha_carrot_asset_audit/report.json
```

`current_validation_ready=true` means the available source frames and local ACT
meshes are present. `full_dataset_policy_audit_ready=true` additionally requires
the learned Octo/ACT checkpoints.

The dataset validator checks all 130 parquet episodes, the 7D state/action
contract, fixed reset and gripper constants, and verifies that the 12 source
images are exact pixels from episode-0 frames `0,30,60,90,120,148`.

MuJoCo validation, when `dm_control` is available:

```bash
conda run -n octo_pt python scripts/validate_aloha_carrot_left_mujoco.py
```

This is the canonical 3D visual validation. It uses the local ALOHA mesh, omits
the right arm and proximal meshes outside the dataset view, runs continuous
warm-start IK, and checks:

- FK reset against the configured initial end-effector position
- six source-aligned overhead/wrist frames
- reset object bbox/centroid and table/backdrop/mat RGB alignment
- wrist wide/close/transfer framing
- end-effector tracking, joint-step, and wrist-camera rotation limits
- pre-grasp/post-place object stability and zero carrot rotation

Generated MuJoCo evidence:

```text
outputs/validation/aloha_carrot_left_mujoco/mujoco_source_timeline_contact_sheet.png
outputs/validation/aloha_carrot_left_mujoco/mujoco_primary_rollout.gif
outputs/validation/aloha_carrot_left_mujoco/mujoco_wrist_rollout.gif
outputs/validation/aloha_carrot_left_mujoco/report.json
```

The timeline validator compares the six available source high/wrist frames with
six semantic simulation frames, checks wrist-view visual probes for the expected
foreground fingers/object/mat/backdrop colors, and writes:

```text
outputs/validation/aloha_carrot_left_timeline/timeline_contact_sheet.png
outputs/validation/aloha_carrot_left_timeline/report.json
```

The visual similarity validator compares source-vs-sim color/centroid features
for the six high/wrist frames and writes:

```text
outputs/validation/aloha_carrot_left_visual_similarity/visual_similarity_contact_sheet.png
outputs/validation/aloha_carrot_left_visual_similarity/report.json
```

## Gym

From the repository root:

```bash
cd examples
conda run -n octo_pt python -c '
import gym
from envs.aloha_sim_env import AlohaGymEnv  # registers aloha-carrot-left-v0

env = gym.make("aloha-carrot-left-v0")
obs, info = env.reset()
print(env.action_space.shape, obs["proprio"].shape, obs["image_primary"].shape)
env.close()
'
```

Autonomous episodes terminate after the left arm retreats from the cup, not at
the first success frame.

## Evidence

Current validation evidence is tracked in:

```text
docs/plans/active/aloha-carrot-simulation-rebuild.md
```

The local parquet dataset is validated and used for reset/timeline calibration.
Learned Octo and ACT checkpoint weights are not present, so autonomous execution
uses the deterministic scripted controller rather than a learned-policy rollout.
