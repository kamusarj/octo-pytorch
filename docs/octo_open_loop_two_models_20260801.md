# Báo cáo open-loop: hai checkpoint Octo ALOHA carrot

Ngày đánh giá: **2026-08-01**
Trạng thái: **hoàn tất**

## 1. Mục tiêu

Báo cáo này đánh giá open-loop hai policy Octo đã fine-tune:

1. `aloha_carrot_w2_h8_stageb_seed42`, checkpoint step `20000`;
2. `one_episode_ep0_jitter1cm_command_v4_adapt`, checkpoint step `2525`.

Mục tiêu là kiểm tra policy có tái tạo được action trong demonstration hay
không, đồng thời đo khoảng cách giữa trajectory train và trajectory validation.
Đây không phải simulation rollout và không đo task success.

## 2. Protocol

Protocol được chuyển thể từ:

- [Isaac-GR00T: Step 4 — Open Loop Evaluation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md#step-4-open-loop-evaluation)
- [NVIDIA `gr00t/eval/open_loop_eval.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/eval/open_loop_eval.py)

Tại mỗi inference point, evaluator:

1. lấy observation history thật từ RLDS trajectory;
2. đưa observation và language instruction vào checkpoint;
3. dự đoán một action chunk;
4. lấy `execution_horizon` action đầu của chunk;
5. ghép các chunk prediction theo thời gian;
6. so sánh với ground-truth action tương ứng;
7. tính MAE, MSE và RMSE trên các action value hợp lệ;
8. lưu plot GT-vs-pred và trace NumPy để audit lại.

Observation luôn là observation ghi sẵn trong dataset (teacher-forced). Action
dự đoán không được đưa ngược vào simulator hoặc dùng để sinh observation tiếp
theo. Vì vậy kết quả chỉ đo chất lượng action imitation.

Metric `original_units` được tính sau khi đảo normalization riêng của từng
checkpoint. Sáu chiều đầu là joint action; chiều cuối là follower-gripper
command. Metric `normalized` được giữ lại để debug normalization, không dùng để
so sánh trực tiếp giữa hai dataset.

## 3. Cấu hình đánh giá

| Model | Checkpoint | Dataset | Window | Action horizon | Execution horizon |
|---|---:|---|---:|---:|---:|
| `aloha_carrot_w2_h8_stageb_seed42` | 20000 | `aloha_carrot_easy_rlds` | 2 | 8 | 8 |
| `one_episode_ep0_jitter1cm_command_v4_adapt` | 2525 | `aloha_carrot_sim_rlds` jitter ±1 cm | 1 | 20 | 20 |

Mỗi model được đánh giá trên trajectory index `0` của hai split:

| Model | Split | Trajectory index | Source episode | Số bước | Số lần inference |
|---|---|---:|---:|---:|---:|
| step 20000 | train | 0 | 0 | 149 | 19 |
| step 20000 | validation | 0 | 6 | 160 | 20 |
| robust step 2525 | train | 0 | 0 | 111 | 6 |
| robust step 2525 | validation | 0 | 24 | 111 | 6 |

`steps=200` được yêu cầu nhưng evaluator tự giới hạn theo chiều dài trajectory.

## 4. Kết quả chính

### 4.1 Metric trong đơn vị action gốc

| Checkpoint | Split | MAE | MSE | RMSE | Gripper accuracy |
|---|---|---:|---:|---:|---:|
| step 20000 | train | 0.028095 | 0.001925 | 0.043876 | 97.99% |
| step 20000 | validation | 0.171922 | 0.097308 | 0.311942 | 96.88% |
| robust step 2525 | train | 0.048855 | 0.006948 | 0.083355 | 99.10% |
| robust step 2525 | validation | 0.055884 | 0.006743 | 0.082118 | 99.10% |

### 4.2 Generalization gap

| Checkpoint | Train MAE | Validation MAE | Chênh lệch | Validation / Train |
|---|---:|---:|---:|---:|
| step 20000 | 0.028095 | 0.171922 | +0.143827 | **6.12×** |
| robust step 2525 | 0.048855 | 0.055884 | +0.007029 | **1.14×** |

### 4.3 MAE theo action dimension

| Checkpoint / split | j0 | j1 | j2 | j3 | j4 | j5 | gripper |
|---|---:|---:|---:|---:|---:|---:|---:|
| step 20000 train | 0.0211 | 0.0352 | 0.0272 | 0.0186 | 0.0304 | 0.0470 | 0.0171 |
| step 20000 validation | 0.0483 | 0.0771 | 0.0552 | **0.4440** | 0.0807 | **0.4663** | 0.0319 |
| robust 2525 train | 0.0453 | 0.0769 | 0.0632 | 0.0103 | 0.0569 | 0.0624 | 0.0270 |
| robust 2525 validation | 0.0524 | 0.0897 | 0.0561 | 0.0370 | 0.0582 | 0.0708 | 0.0270 |

### 4.4 Metric normalized để audit

| Checkpoint | Split | Normalized MAE | Normalized MSE |
|---|---|---:|---:|
| step 20000 | train | 0.105843 | 0.036207 |
| step 20000 | validation | 0.889350 | 3.408159 |
| robust step 2525 | train | 0.154744 | 0.065366 |
| robust step 2525 | validation | 0.210640 | 0.150732 |

## 5. Diễn giải

### Checkpoint step 20000

Prediction trên train episode 00 bám ground truth khá sát. Tuy nhiên validation
MAE cao hơn train `6.12×`. Sai số tập trung chủ yếu tại joint 3 và joint 5,
với MAE lần lượt `0.4440` và `0.4663`. Gripper vẫn dự đoán đúng trạng thái phần
lớn thời gian, nên gripper accuracy cao không bù được lỗi joint trajectory.

Kết quả này là bằng chứng rõ rằng checkpoint step 20000 đã fit tốt train
trajectory nhưng tổng quát hóa kém sang held-out episode 6.

### Checkpoint robust step 2525

Validation MAE chỉ cao hơn train khoảng `14.4%`. Không action dimension nào có
mức tăng đột biến như checkpoint step 20000. Điều này phù hợp với mục tiêu của
fine-tune jitter ±1 cm: chấp nhận train error cao hơn một chút để prediction ổn
định hơn trên layout held-out cùng distribution.

### Không dùng metric này để suy ra success rate

Open-loop không mô phỏng state drift, contact, grasp failure hoặc recovery. Một
policy có MAE thấp vẫn có thể thất bại closed-loop; ngược lại một policy có một
số sai lệch joint nhưng vẫn hoàn thành task nhờ feedback/replanning. Kết quả
closed-loop `8/10` của checkpoint robust là phép đo độc lập, không được trộn vào
MAE/MSE trong báo cáo này.

Hai hàng checkpoint cũng không phải so sánh tuyệt đối hoàn toàn công bằng vì
chúng sử dụng dataset, trajectory, normalization và action horizon khác nhau.
Tín hiệu đáng tin nhất trong báo cáo này là **generalization gap bên trong từng
model**.

## 6. Artifact được tạo

Kết quả local nằm dưới:

```text
outputs/eval/open_loop_two_models_20260801/
├── aloha_carrot_w2_h8_stageb_step20000/
│   ├── summary.json
│   ├── trajectory_metrics.csv
│   ├── train/trajectory_000_gt_vs_pred.png
│   ├── train/trajectory_000_actions.npz
│   ├── validation/trajectory_000_gt_vs_pred.png
│   └── validation/trajectory_000_actions.npz
└── one_episode_ep0_jitter1cm_command_v4_step2525/
    ├── summary.json
    ├── trajectory_metrics.csv
    ├── train/trajectory_000_gt_vs_pred.png
    ├── train/trajectory_000_actions.npz
    ├── validation/trajectory_000_gt_vs_pred.png
    └── validation/trajectory_000_actions.npz
```

Các artifact trong `outputs/` bị Git ignore; báo cáo này ghi lại toàn bộ metric
cần thiết còn JSON, CSV, plot và NumPy trace được giữ local để audit.

## 7. Cách tái chạy

Chạy đúng hai checkpoint, tuần tự để tránh host-RAM OOM:

```bash
PYTHON_BIN=/home/linh/anaconda3/envs/octo_pt/bin/python \
OUTPUT_ROOT=outputs/eval/open_loop_two_models_20260801 \
SPLITS=train,validation \
TRAJ_IDS=0 \
STEPS=200 \
DEVICE=cuda:0 \
bash scripts/run_octo_two_model_open_loop.sh
```

Chạy một checkpoint tùy ý:

```bash
/home/linh/anaconda3/envs/octo_pt/bin/python \
  scripts/evaluate_octo_open_loop.py \
  --checkpoint-dir checkpoints/octo/aloha_carrot_w2_h8_stageb_seed42 \
  --checkpoint-step 20000 \
  --splits train,validation \
  --traj-ids 0 \
  --steps 200 \
  --execution-horizon 0 \
  --device cuda:0 \
  --output-dir outputs/eval/open_loop_step20000
```

`--execution-horizon 0` tự lấy full action horizon từ checkpoint. Có thể truyền
giá trị nhỏ hơn để đo protocol replanning dày hơn, nhưng không được lớn hơn
action horizon của model.

## 8. Kiểm tra kỹ thuật

- evaluator đã chạy end-to-end trên cả hai checkpoint;
- tất cả trace có shape `(T, 7)` cho prediction, GT và valid mask;
- checkpoint step 20000 dùng 19/20 inference call cho train/validation;
- checkpoint robust dùng 6 inference call trên mỗi trajectory;
- `python -m unittest tests.test_train_utils_pt` chạy thành công;
- Python compile, Bash syntax và `git diff --check` đều thành công;
- hai model được load tuần tự, không đồng thời chiếm RAM/GPU.
