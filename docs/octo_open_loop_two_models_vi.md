# Báo cáo open-loop: hai checkpoint Octo ALOHA carrot trên dữ liệu gốc

Ngày đánh giá lại: **2026-08-03**
Trạng thái: **hoàn tất**

## 1. Mục tiêu

Báo cáo này đánh giá open-loop hai policy Octo đã fine-tune:

1. `aloha_carrot_w2_h8_stageb_seed42`, checkpoint step `20000`;
2. `one_episode_ep0_jitter1cm_command_v4_adapt`, checkpoint step `2525`.

Mục tiêu là kiểm tra policy có tái tạo được action trong demonstration hay
không, đồng thời đo khoảng cách giữa trajectory train và trajectory validation.
Cả hai checkpoint được đánh giá trên cùng trajectory gốc và cùng tần suất
inference để kết quả có thể so sánh. Đây không phải simulation rollout và
không đo task success.

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
so sánh trực tiếp giữa hai checkpoint vì chúng dùng statistics khác nhau.

## 3. Cấu hình đánh giá

| Model | Checkpoint | Dataset đánh giá | Window | Action horizon | Execution horizon |
|---|---:|---|---:|---:|---:|
| `aloha_carrot_w2_h8_stageb_seed42` | 20000 | `aloha_carrot_easy_rlds` | 2 | 8 | 8 |
| `one_episode_ep0_jitter1cm_command_v4_adapt` | 2525 | `aloha_carrot_easy_rlds` | 1 | 20 | 8 |

Dataset đánh giá của cả hai model là cùng một bản RLDS gốc tại
`outputs/derived/aloha_carrot_easy_rlds`. Checkpoint step 20000 dùng
normalization statistics của toàn bộ train split; checkpoint step 2525 giữ
normalization statistics của episode 0 đúng theo contract lúc fine-tune. Mọi
metric dùng để so sánh bên dưới đều ở `original_units`.

Mỗi model được đánh giá trên trajectory index `0` của hai split:

| Model | Split | Trajectory index | Source episode | Số bước | Số lần inference |
|---|---|---:|---:|---:|---:|
| step 20000 | train | 0 | 0 | 149 | 19 |
| step 20000 | validation | 0 | 6 | 160 | 20 |
| jitter-adapted step 2525 | train | 0 | 0 | 149 | 19 |
| jitter-adapted step 2525 | validation | 0 | 6 | 160 | 20 |

`steps=200` được yêu cầu nhưng evaluator tự giới hạn theo chiều dài trajectory.

## 4. Kết quả chính

### 4.1 Metric trong đơn vị action gốc

| Checkpoint | Split | MAE | MSE | RMSE | Gripper accuracy |
|---|---|---:|---:|---:|---:|
| step 20000 | train | 0.028095 | 0.001925 | 0.043876 | 97.99% |
| step 20000 | validation | 0.171922 | 0.097308 | 0.311942 | 96.88% |
| jitter-adapted step 2525 | train | 0.280125 | 0.145849 | 0.381901 | 67.11% |
| jitter-adapted step 2525 | validation | 0.353794 | 0.205040 | 0.452813 | 70.00% |

### 4.2 Generalization gap

| Checkpoint | Train MAE | Validation MAE | Chênh lệch | Validation / Train |
|---|---:|---:|---:|---:|
| step 20000 | 0.028095 | 0.171922 | +0.143827 | **6.12×** |
| jitter-adapted step 2525 | 0.280125 | 0.353794 | +0.073669 | **1.26×** |

### 4.3 MAE theo action dimension

| Checkpoint / split | j0 | j1 | j2 | j3 | j4 | j5 | gripper |
|---|---:|---:|---:|---:|---:|---:|---:|
| step 20000 train | 0.0211 | 0.0352 | 0.0272 | 0.0186 | 0.0304 | 0.0470 | 0.0171 |
| step 20000 validation | 0.0483 | 0.0771 | 0.0552 | **0.4440** | 0.0807 | **0.4663** | 0.0319 |
| jitter-adapted 2525 train | 0.2422 | **0.5394** | 0.4049 | 0.0828 | 0.3126 | 0.2297 | 0.1494 |
| jitter-adapted 2525 validation | 0.2375 | 0.4469 | 0.3273 | **0.5432** | 0.2700 | **0.5104** | 0.1413 |

### 4.4 Metric normalized để audit

| Checkpoint | Split | Normalized MAE | Normalized MSE |
|---|---|---:|---:|
| step 20000 | train | 0.105843 | 0.036207 |
| step 20000 | validation | 0.889350 | 3.408159 |
| jitter-adapted step 2525 | train | 0.871781 | 1.123849 |
| jitter-adapted step 2525 | validation | 1.763214 | 11.760666 |

### 4.5 Biểu đồ GT-vs-prediction

Các đường màu xanh dương là ground-truth action, các đường màu cam là action dự
đoán và đường màu xám nét đứt là state tham chiếu. Chấm tròn cùng vạch dọc màu
tím đánh dấu thời điểm inference. Ngoài màu sắc tương phản, kiểu nét và marker
khác nhau giúp biểu đồ vẫn dễ đọc khi in thang xám hoặc với người bị rối loạn
sắc giác.

**Checkpoint step 20000 — train trajectory 0**

![Checkpoint step 20000, train trajectory 0: ground-truth action so với predicted action](assets/open_loop/step20000_train_gt_vs_pred.png)

**Checkpoint step 20000 — validation trajectory 0**

![Checkpoint step 20000, validation trajectory 0: ground-truth action so với predicted action](assets/open_loop/step20000_validation_gt_vs_pred.png)

**Checkpoint jitter-adapted step 2525 — train trajectory 0**

![Checkpoint jitter-adapted step 2525, train trajectory 0: ground-truth action so với predicted action](assets/open_loop/jitter_adapted_step2525_train_gt_vs_pred.png)

**Checkpoint jitter-adapted step 2525 — validation trajectory 0**

![Checkpoint jitter-adapted step 2525, validation trajectory 0: ground-truth action so với predicted action](assets/open_loop/jitter_adapted_step2525_validation_gt_vs_pred.png)

## 5. Diễn giải

### Checkpoint step 20000

Prediction trên train episode 00 bám ground truth khá sát. Tuy nhiên validation
MAE cao hơn train `6.12×`. Sai số tập trung chủ yếu tại joint 3 và joint 5,
với MAE lần lượt `0.4440` và `0.4663`. Gripper vẫn dự đoán đúng trạng thái phần
lớn thời gian, nên gripper accuracy cao không bù được lỗi joint trajectory.

Kết quả này là bằng chứng rõ rằng checkpoint step 20000 đã fit tốt train
trajectory nhưng tổng quát hóa kém sang held-out episode 6.

### Checkpoint jitter-adapted step 2525

Trên đúng episode 0 gốc, checkpoint này có MAE `0.280125`, gần `10×` MAE của
checkpoint step 20000. Validation MAE là `0.353794`, khoảng `2.06×` checkpoint
step 20000 trên cùng episode 6. Gripper accuracy cũng chỉ đạt `67.11%` trên
train và `70.00%` trên validation.

Generalization ratio `1.26×` không có nghĩa checkpoint này tốt hơn: train error
đã rất cao. Plot cho thấy prediction lệch mạnh trên nhiều joint ngay cả với
episode 0. Kết quả chỉ chứng minh checkpoint jitter-adapted không tái tạo tốt
action của dataset gốc theo protocol open-loop này; không được thay bằng metric
từ dataset mô phỏng.

### Không dùng metric này để suy ra success rate

Open-loop không mô phỏng state drift, contact, grasp failure hoặc recovery. Một
policy có MAE thấp vẫn có thể thất bại closed-loop; ngược lại một policy có một
số sai lệch joint nhưng vẫn hoàn thành task nhờ feedback/replanning. Kết quả
closed-loop là phép đo độc lập, không được trộn vào MAE/MSE trong báo cáo này.

Hai checkpoint dùng cùng observation, ground-truth action và
`execution_horizon=8`; trace GT giữa hai run sai khác tối đa dưới `1.2e-7` do
làm tròn float. Chúng vẫn có window, action horizon và normalization contract
khác nhau, nên cần đọc cả metric lẫn plot thay vì coi đây là so sánh kiến trúc
tuyệt đối.

## 6. Artifact được tạo

Kết quả local nằm dưới:

```text
outputs/eval/open_loop_two_models_original_data/
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

Các artifact đầy đủ trong `outputs/` bị Git ignore; báo cáo này ghi lại toàn bộ
metric cần thiết, còn JSON, CSV và NumPy trace được giữ local để audit. Bốn plot
GT-vs-prediction ở mục 4.5 được lưu trong `docs/assets/open_loop/` và commit vào
Git để hiển thị trực tiếp trên GitHub.

## 7. Cách tái chạy

Chạy đúng hai checkpoint, tuần tự để tránh host-RAM OOM:

```bash
PYTHON_BIN=/home/linh/anaconda3/envs/octo_pt/bin/python \
OUTPUT_ROOT=outputs/eval/open_loop_two_models_original_data \
DATA_DIR=outputs/derived/aloha_carrot_easy_rlds \
DATASET_NAME=aloha_carrot_easy_rlds \
SPLITS=train,validation \
TRAJ_IDS=0 \
STEPS=200 \
EXECUTION_HORIZON=8 \
DEVICE=cuda:0 \
bash scripts/run_octo_two_model_open_loop.sh
```

Chạy riêng checkpoint jitter-adapted trên dữ liệu gốc:

```bash
/home/linh/anaconda3/envs/octo_pt/bin/python \
  scripts/evaluate_octo_open_loop.py \
  --checkpoint-dir checkpoints/octo/one_episode_ep0_jitter1cm_command_v4_adapt \
  --checkpoint-step 2525 \
  --data-dir outputs/derived/aloha_carrot_easy_rlds \
  --dataset-name aloha_carrot_easy_rlds \
  --splits train,validation \
  --traj-ids 0 \
  --steps 200 \
  --execution-horizon 8 \
  --device cuda:0 \
  --output-dir outputs/eval/open_loop_jitter_adapted_original_data
```

`execution_horizon=8` được dùng chung để hai checkpoint có cùng inference
points. Checkpoint step 2525 vẫn dự đoán action chunk dài 20 nhưng evaluator chỉ
lấy 8 action đầu trước lần inference tiếp theo.

## 8. Kiểm tra kỹ thuật

- evaluator đã chạy end-to-end trên cả hai checkpoint;
- tất cả trace có shape `(T, 7)` cho prediction, GT và valid mask;
- checkpoint step 20000 dùng 19/20 inference call cho train/validation;
- checkpoint jitter-adapted dùng 19/20 inference call cho train/validation;
- Parquet gốc và RLDS đều xác nhận train episode 0 có 149 bước, validation
  episode 6 có 160 bước;
- ground-truth trace của hai checkpoint có cùng shape, valid mask giống hệt và
  sai khác tối đa dưới `1.2e-7`;
- kết quả 111 bước trước đây thuộc
  `aloha_carrot_ep0_jitter1cm_command_v4_npz/episode_000000.npz`, là trajectory
  mô phỏng dừng khi reward đạt success; kết quả đó đã bị loại khỏi báo cáo này;
- `python -m unittest tests.test_train_utils_pt` chạy thành công;
- Python compile, Bash syntax và `git diff --check` đều thành công;
- hai model được load tuần tự, không đồng thời chiếm RAM/GPU.
