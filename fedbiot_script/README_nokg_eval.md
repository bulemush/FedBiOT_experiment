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
data/CWQ/ComplexWebQuestions_train.json
data/CWQ/ComplexWebQuestions_dev.json
data/CWQ/ComplexWebQuestions_test.json
data/GraphQuestions/graphquestions.training.json
data/GraphQuestions/graphquestions.testing.json
data/kqa_pro/train.json
data/kqa_pro/val.json
data/kqa_pro/test.json
data/openbookQA/main/train-00000-of-00001.parquet
data/openbookQA/main/validation-00000-of-00001.parquet
data/openbookQA/main/test-00000-of-00001.parquet
```

NoKG YAML 已在 `data.args` 中显式指定这些本地文件。数据读取支持 `json`、`jsonl`、`parquet`，也支持 HuggingFace `load_from_disk` 保存的目录。`GraphQuestions` 和 `kqa_pro` 在这里只读取 `question` 与 `answer` 字段，不构造 `sg` 或 `kg_inputs`。`OpenBookQA` 使用 `openbookqa_mcqa@llm`，不使用 ConceptNet 预处理，也不会在本地 NoKG 配置中回退到 HuggingFace 下载。

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

建议按“一个方法 + 一个数据集”分别启动训练。这样每个 nohup 任务只负责一个数据集的 3 个 seed；如果某个数据集 OOM、断开或报错，不会影响其他数据集。

两组实验仍然保持相同的 backbone、LoRA 参数、客户端数量、数据划分、batch size、local update steps、optimizer/lr、rounds 和 seeds。当前 YAML 已加入模型并行显存约束：

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

### 单数据集训练脚本

训练统一使用 `fedbiot_script/train_nokg_one.sh`。常用变量如下：

```bash
METHOD=fedbiot        # fedbiot 或 fedot
DATASET=cwq          # cwq / graphquestions / kqapro / openbookqa
SEEDS="1 2 3"
GPU=0,1              # 双卡训练
BATCH_SIZE=2
```

每个 seed 的详细日志会写入 `logs/nokg/train_${METHOD}_${DATASET}_seed${seed}.log`，driver 日志只记录当前数据集整体进度。

### FedBiOT-NoKG 逐数据集训练

```bash
mkdir -p logs/nokg checkpoints/nokg/fedbiot

METHOD=fedbiot DATASET=cwq SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedbiot_cwq_driver.log 2>&1 &

METHOD=fedbiot DATASET=graphquestions SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedbiot_graphquestions_driver.log 2>&1 &

METHOD=fedbiot DATASET=kqapro SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedbiot_kqapro_driver.log 2>&1 &

METHOD=fedbiot DATASET=openbookqa SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedbiot_openbookqa_driver.log 2>&1 &
```

建议一次只启动其中一条。确认完成后再启动下一个数据集：

```bash
tail -f logs/nokg/train_fedbiot_cwq_driver.log
tail -f logs/nokg/train_fedbiot_cwq_seed1.log
```

### FedOT-NoKG 逐数据集训练

```bash
mkdir -p logs/nokg checkpoints/nokg/fedot

METHOD=fedot DATASET=cwq SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedot_cwq_driver.log 2>&1 &

METHOD=fedot DATASET=graphquestions SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedot_graphquestions_driver.log 2>&1 &

METHOD=fedot DATASET=kqapro SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedot_kqapro_driver.log 2>&1 &

METHOD=fedot DATASET=openbookqa SEEDS="1 2 3" GPU=0,1 \
nohup bash fedbiot_script/train_nokg_one.sh > logs/nokg/train_fedot_openbookqa_driver.log 2>&1 &
```

### Smoke Training

只做快速连通性测试时，对单个方法和单个数据集覆盖 rounds、local steps：

```bash
METHOD=fedbiot DATASET=cwq SEEDS=1 GPU=0,1 TOTAL_ROUNDS=1 LOCAL_STEPS=1 ALIGN_STEPS=1 \
bash fedbiot_script/train_nokg_one.sh

METHOD=fedot DATASET=cwq SEEDS=1 GPU=0,1 TOTAL_ROUNDS=1 LOCAL_STEPS=1 ALIGN_STEPS=1 \
bash fedbiot_script/train_nokg_one.sh
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

训练完成后，可以先检查当前 checkpoint 是否齐全：

```bash
python fedbiot_script/preflight_nokg_eval.py
```

评估只使用单卡。每次只评估一个方法的一个数据集，结果写入 `results/nokg_eval/${method}/${dataset}/seed${seed}/`。

### FedBiOT-NoKG 逐数据集评估

```bash
mkdir -p logs/nokg_eval

GPU=0 METHODS=fedbiot DATASETS=cwq SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedbiot_cwq_driver.log 2>&1 &

GPU=0 METHODS=fedbiot DATASETS=graphquestions SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedbiot_graphquestions_driver.log 2>&1 &

GPU=0 METHODS=fedbiot DATASETS=kqapro SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedbiot_kqapro_driver.log 2>&1 &

GPU=0 METHODS=fedbiot DATASETS=openbookqa SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedbiot_openbookqa_driver.log 2>&1 &
```

### FedOT-NoKG 逐数据集评估

```bash
mkdir -p logs/nokg_eval

GPU=0 METHODS=fedot DATASETS=cwq SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedot_cwq_driver.log 2>&1 &

GPU=0 METHODS=fedot DATASETS=graphquestions SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedot_graphquestions_driver.log 2>&1 &

GPU=0 METHODS=fedot DATASETS=kqapro SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedot_kqapro_driver.log 2>&1 &

GPU=0 METHODS=fedot DATASETS=openbookqa SEEDS="1 2 3" \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedot_openbookqa_driver.log 2>&1 &
```

### Smoke Evaluation

只快速评估一个 seed 和少量样本时：

```bash
mkdir -p logs/nokg_eval_smoke

GPU=0 METHODS=fedbiot DATASETS=cwq SEEDS=1 LIMIT=5 OUTDIR=results/nokg_eval_smoke LOGDIR=logs/nokg_eval_smoke \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval_smoke/eval_fedbiot_cwq_driver.log 2>&1 &
```

断点续评时加上 `SKIP_EXISTING=1`：

```bash
GPU=0 METHODS=fedbiot DATASETS=cwq SEEDS="1 2 3" SKIP_EXISTING=1 \
nohup bash fedbiot_script/eval_nokg_all.sh > logs/nokg_eval/eval_fedbiot_cwq_resume.log 2>&1 &
```

所有数据集评估完成后再汇总与检查：

```bash
python fedbiot_script/collect_nokg_eval.py --indir results/nokg_eval
python fedbiot_script/check_nokg_results.py
```

正式实验矩阵为：

```text
2 methods x 4 datasets x 3 seeds = 24 runs
```
