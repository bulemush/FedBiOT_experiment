# FedBiOT/FedOT NoKG 基线实验说明

本目录只用于运行 `FedBiOT` 和 `FedOT` 的无知识图谱基线实验。这里不实现、不配置、不评估 FedKBS；FedKBS 结果由另一个项目产生，本项目输出的结果用于外部对比。

四个任务为：

```text
CWQ
GraphQuestions
kqa_pro
OpenBookQA
```

所有 NoKG YAML 均满足：

```yaml
federate:
  total_round_num: 200
llm:
  model_parallel:
    use: true
    device_map: balanced_layers
    max_memory:
      '0': 14GiB
      '1': 22GiB
  kg_adapter:
    use: false
dataloader:
  batch_size: 2
```

FedBiOT 与 FedOT 的唯一区别是：

```yaml
# FedBiOT-NoKG
llm.offsite_tuning.emu_align.initial_only: true

# FedOT-NoKG
llm.offsite_tuning.emu_align.initial_only: false
```

## 数据放置

推荐将数据放在 `data/` 下，示例结构如下：

```text
data/cwq/{train,validation,test}.jsonl
data/GraphQuestions/{train,validation,test}.jsonl
data/kqa_pro/{train,validation,test}.jsonl
data/openbookQA/main/{train,validation,test}.parquet
```

数据读取支持 `json`、`jsonl`、`parquet`，也支持 HuggingFace `load_from_disk` 保存的目录。`GraphQuestions` 和 `kqa_pro` 在这里只读取 `question` 与 `answer` 字段，不构造 `sg` 或 `kg_inputs`。`OpenBookQA` 使用 `openbookqa_mcqa@llm`，不使用 ConceptNet 预处理。

## 静态检查

先检查 YAML 和 NoKG 约束：

```bash
python fedbiot_script/verify_nokg_eval_setup.py
```

训练前检查数据是否能被读取：

```bash
python fedbiot_script/preflight_nokg_eval.py --skip-checkpoints
```

## 训练

建议将 FedBiOT 和 FedOT 分开训练。这样更方便单独观察 OOM、断点、日志和耗时，也避免一个方法失败时影响另一个方法的运行队列。

两组实验仍然保持相同的 backbone、LoRA 参数、客户端数量、数据划分、batch size、local update steps、optimizer/lr、rounds 和 seeds。

当前 YAML 已加入模型并行显存约束：

```yaml
llm:
  model_parallel:
    use: true
    device_map: balanced_layers
    max_memory:
      '0': 14GiB
      '1': 22GiB
    same_device_map: false
```

这会让模型层按可用显存比例分配到双卡上，减少单卡显存峰值。

### 训练 FedBiOT-NoKG

双卡训练，使用 `CUDA_VISIBLE_DEVICES=0,1`：

```bash
mkdir -p logs/nokg checkpoints/nokg/fedbiot

nohup bash -lc '
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

method="fedbiot"
for ds in cwq graphquestions kqapro openbookqa; do
  for seed in 1 2 3; do
    cfg="fedbiot_script/${method}_nokg/${ds}.yaml"
    save_to="checkpoints/nokg/${method}/${ds}_seed${seed}.ckpt"
    expname="${method}_nokg/${ds}_seed${seed}"
    log="logs/nokg/train_${method}_${ds}_seed${seed}.log"

    echo "[START] ${method} ${ds} seed=${seed}"
    python federatedscope/main.py --cfg "$cfg" \
      seed "$seed" \
      device 0 \
      dataloader.batch_size 2 \
      federate.save_to "$save_to" \
      expname "$expname" \
      > "$log" 2>&1
    echo "[DONE] ${method} ${ds} seed=${seed}"
  done
done
' > logs/nokg/train_fedbiot_driver.log 2>&1 &
```

查看 FedBiOT 训练进度：

```bash
tail -f logs/nokg/train_fedbiot_driver.log
```

### 训练 FedOT-NoKG

FedOT 单独启动：

```bash
mkdir -p logs/nokg checkpoints/nokg/fedot

nohup bash -lc '
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

method="fedot"
for ds in cwq graphquestions kqapro openbookqa; do
  for seed in 1 2 3; do
    cfg="fedbiot_script/${method}_nokg/${ds}.yaml"
    save_to="checkpoints/nokg/${method}/${ds}_seed${seed}.ckpt"
    expname="${method}_nokg/${ds}_seed${seed}"
    log="logs/nokg/train_${method}_${ds}_seed${seed}.log"

    echo "[START] ${method} ${ds} seed=${seed}"
    python federatedscope/main.py --cfg "$cfg" \
      seed "$seed" \
      device 0 \
      dataloader.batch_size 2 \
      federate.save_to "$save_to" \
      expname "$expname" \
      > "$log" 2>&1
    echo "[DONE] ${method} ${ds} seed=${seed}"
  done
done
' > logs/nokg/train_fedot_driver.log 2>&1 &
```

查看 FedOT 训练进度：

```bash
tail -f logs/nokg/train_fedot_driver.log
```

### Smoke Training

只做快速连通性测试时，可以覆盖 rounds 和 local steps：

```bash
python federatedscope/main.py --cfg fedbiot_script/fedbiot_nokg/cwq.yaml \
  seed 1 \
  device 0 \
  federate.total_round_num 1 \
  train.local_update_steps 1 \
  llm.offsite_tuning.emu_align.train.local_update_steps 1
```

FedOT 的 smoke run：

```bash
python federatedscope/main.py --cfg fedbiot_script/fedot_nokg/cwq.yaml \
  seed 1 \
  device 0 \
  federate.total_round_num 1 \
  train.local_update_steps 1 \
  llm.offsite_tuning.emu_align.train.local_update_steps 1
```

## OOM 处理建议

如果训练中出现 OOM，优先保证 FedBiOT 和 FedOT 的改动完全一致。推荐按顺序尝试：

```bash
dataloader.batch_size 1
llm.tok_len 512
llm.grad_accum_step 4
```

示例：

```bash
python federatedscope/main.py --cfg "$cfg" \
  seed "$seed" \
  device 0 \
  dataloader.batch_size 1 \
  llm.tok_len 512 \
  llm.grad_accum_step 4 \
  federate.save_to "$save_to" \
  expname "$expname"
```

不要只降低 FedOT 或只降低某一个数据集的配置；如果正式实验需要降显存，24 个主实验应统一使用同一套降显存配置。

## 评估

训练完成后，先检查 checkpoint 是否齐全：

```bash
python fedbiot_script/preflight_nokg_eval.py
```

评估只使用单卡。Smoke 评估：

```bash
mkdir -p logs/nokg_eval_smoke
GPU=0 SEEDS=1 LIMIT=5 OUTDIR=results/nokg_eval_smoke LOGDIR=logs/nokg_eval_smoke \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval_smoke_driver.log 2>&1 &
```

完整评估：

```bash
mkdir -p logs/nokg_eval
GPU=0 nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval_driver.log 2>&1 &
```

只评估 FedBiOT：

```bash
GPU=0 METHODS=fedbiot nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval_fedbiot.log 2>&1 &
```

只评估 FedOT：

```bash
GPU=0 METHODS=fedot nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval_fedot.log 2>&1 &
```

断点续评：

```bash
GPU=0 SKIP_EXISTING=1 nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval_resume.log 2>&1 &
```

汇总与检查：

```bash
python fedbiot_script/collect_nokg_eval.py --indir results/nokg_eval
python fedbiot_script/check_nokg_results.py
```

正式实验矩阵为：

```text
2 methods x 4 datasets x 3 seeds = 24 runs
```
