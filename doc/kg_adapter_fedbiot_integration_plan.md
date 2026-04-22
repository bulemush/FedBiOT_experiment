# KG-adapter 与 FedBiOT 深度结合代码修改方案

## 1. 目标

在现有 FedBiOT 的基础上，将另一篇论文中的 KG-adapter 作为结构化知识增强模块，插入到原本用于 adapter 的最后几层中，并继续复用当前仓库已经具备的：

- `offsite_tuning` 的 emulator/adapter 切分机制
- `AdapterModel.state_dict()` 只返回可训练参数的传输机制
- LoRA/PEFT 的低秩训练能力
- 现有客户端训练、服务端聚合、断点恢复、评测链路

最终形成下面的训练范式：

1. 基础 LLM 主干保持冻结
2. 中间 emulator 继续由 FedBiOT/offsite-tuning 负责
3. adapter 区域的最后若干层替换为 `KGAdapterAugmentedLayer`
4. `KGAdapterAugmentedLayer` 内部再挂接 KG-adapter 模块
5. KG-adapter 内部的线性层用 LoRA 包装，只在联邦过程中同步 LoRA 权重

## 2. 当前 FedBiOT 现状

从当前仓库代码看，已经具备和本方案高度匹配的 3 个关键前提：

### 2.1 模型构建侧

- `federatedscope/llm/model/model_builder.py`
  - 负责加载 HuggingFace LLM
  - 最后统一包装成 `AdapterModel`
- `federatedscope/llm/model/adapter_builder.py`
  - `AdapterModel.state_dict(return_trainable=True)` 默认只返回可训练参数
  - 对 PEFT/LoRA 已经有原生支持

### 2.2 Offsite-tuning 切层侧

- `federatedscope/llm/offsite_tuning/utils.py`
  - `generate_emulator_and_adapter()`
  - `set_layers()`
  - `convert_layers_train_state()`

当前 FedBiOT 的“adapter”本质上是：

- emulator 左边的前置层
- emulator 右边的后置层

也就是说，你要插入 KG-adapter 的最佳位置，不是整个模型任意层，而是：

- `new_model.adapter` 所代表的那一段层集合中的“最后几层”

### 2.3 联邦训练与聚合侧

- `federatedscope/core/trainers/torch_trainer.py`
  - `get_model_para()` 上传的是 `ctx.model.state_dict()` 过滤后的参数
- `federatedscope/llm/llm_local/aggregator.py`
  - 已支持按 key 做 LoRA 参数平均
- `federatedscope/llm/offsite_tuning/server.py`
  - 将 emulator+adapter 广播给客户端
- `federatedscope/llm/offsite_tuning/client.py`
  - 客户端接收模型后复用同一 trainer 流程

因此，本次改造最核心的思路不是重写联邦框架，而是：

- 让 KG-adapter 成为 `AdapterModel` 内的一个 trainable 子模块
- 让 `requires_grad=True` 的参数严格限制为 KG-adapter 的 LoRA 参数
- 复用当前 trainer / client / server / aggregator 的已有联邦链路

## 3. 推荐的整体结构

建议把两篇论文的结合方式实现成下面这条主路径：

```text
Raw LLM
  -> AdapterModel
  -> OffsiteTuning.generate_emulator_and_adapter()
  -> adapter 区域最后 K 层替换为 KGAdapterAugmentedLayer
  -> 仅开放 KG-adapter 中的 LoRA 参数训练
  -> client 本地更新
  -> server 仅聚合 KG LoRA 参数
```

建议区分 3 类参数：

### 3.1 全局冻结参数

- 原始 LLM backbone
- emulator 层
- adapter 区域中未插入 KG-adapter 的普通层
- KG 图谱编码器中不希望跨客户端同步的静态投影层

### 3.2 客户端本地参数

适合放本地不聚合的内容：

- client-specific KG cache
- 实体/关系索引映射
- 局部图谱检索缓存
- 可选的 prompt router / retrieval memory

### 3.3 联邦同步参数

建议只同步：

- KG-adapter 内各线性层的 LoRA A/B
- 可选的 gate / fusion scalar / layernorm 偏置

不建议直接同步：

- 整个 KG-adapter 全量权重
- 实体 embedding 表
- 大体量图结构张量

## 4. 代码改造分层设计

## 4.1 新增 KG-adapter 模块目录

建议新增目录：

- `federatedscope/llm/kg_adapter/`

建议文件：

- `federatedscope/llm/kg_adapter/__init__.py`
- `federatedscope/llm/kg_adapter/modules.py`
- `federatedscope/llm/kg_adapter/layer.py`
- `federatedscope/llm/kg_adapter/utils.py`

推荐职责如下：

### `modules.py`

定义最小可插拔 KG-adapter 组件：

- `KGAdapterConfig`
- `KGEncoder`
- `KGCrossAttention` 或 `KGMessageFusion`
- `KGFusionGate`
- `LoRAWrappedLinear`

### `layer.py`

定义适配任意 decoder layer 的包装器：

- `KGAdapterAugmentedLayer`

它接收一个原始 decoder layer，然后在 `forward()` 中执行：

1. 原始 layer 前向
2. 取 hidden states
3. 调用 KG-adapter 生成知识增量
4. residual/gated fusion 回 hidden states

伪代码如下：

```python
class KGAdapterAugmentedLayer(nn.Module):
    def __init__(self, base_layer, kg_adapter, fusion_mode="residual"):
        super().__init__()
        self.base_layer = base_layer
        self.kg_adapter = kg_adapter
        self.fusion_mode = fusion_mode

    def forward(self, *args, **kwargs):
        output = self.base_layer(*args, **kwargs)
        hidden = output[0] if isinstance(output, tuple) else output
        kg_delta = self.kg_adapter(hidden, **self._extract_kg_inputs(kwargs))
        fused = hidden + kg_delta
        if isinstance(output, tuple):
            return (fused,) + output[1:]
        return fused
```

## 4.2 扩展 `AdapterModel`

修改文件：

- `federatedscope/llm/model/adapter_builder.py`

建议新增能力：

### A. 挂载 KG-adapter

新增方法：

- `attach_kg_adapters(...)`
- `get_federated_trainable_state_dict()`
- `mark_only_kg_lora_trainable()`

作用：

- 在 adapter 区域最后几层中插入 KG 包装层
- 统一控制哪些参数参与联邦通信

### B. 更细粒度的 trainable 参数筛选

当前 `get_trainable_state_dict()` 的逻辑是：

- 只要 `requires_grad=True` 就会被带上

建议扩展成双层过滤：

1. `requires_grad=True`
2. 参数名命中 `llm.kg_adapter.federate_param_keywords`

例如：

```python
def get_federated_trainable_state_dict(self):
    patterns = getattr(self, "federate_param_keywords", None)
    state_dict = self.model.state_dict()
    trainable = OrderedDict()
    for name, param in self.model.named_parameters():
        if not param.requires_grad:
            continue
        if patterns and not any(p in name for p in patterns):
            continue
        trainable[name] = state_dict[name]
    return trainable
```

然后让 `state_dict(return_trainable=True)` 在启用 KG-adapter 时优先返回这个版本。

这样做的好处是：

- 即使某些 gate 或 norm 被训练，也能决定是否联邦同步
- 可以避免误把 emulator 或普通 adapter 权重上传到服务端

## 4.3 在模型构建阶段插入 KG-adapter

修改文件：

- `federatedscope/llm/model/model_builder.py`

建议在 `get_llm()` 中增加：

1. 正常加载基础模型
2. 正常包一层 `AdapterModel`
3. 如果 `cfg.llm.kg_adapter.use=True`
   - 延后到 `offsite_tuning` 切层之后再插入 KG-adapter

这里要注意：

- KG-adapter 应该插在 `generate_emulator_and_adapter()` 之后
- 因为只有那时我们才知道 adapter 区域究竟是哪几层

所以更推荐把真正的插入逻辑放到：

- `federatedscope/llm/offsite_tuning/utils.py`

而 `model_builder.py` 只负责把 KG 相关配置带下去。

## 4.4 在 offsite-tuning 组装阶段注入 KG-adapter

修改文件：

- `federatedscope/llm/offsite_tuning/utils.py`

这是本次改造最关键的文件。

建议在 `generate_emulator_and_adapter()` 的后半段，`new_model = set_layers(...)` 之后，增加：

- `inject_kg_adapter_into_final_layers(new_model, cfg.llm.kg_adapter)`

建议新增函数：

- `inject_kg_adapter_into_final_layers(adapter_model, kg_cfg)`
- `get_adapter_layer_indices(adapter_model)`
- `freeze_non_kg_trainable_params(adapter_model, kg_cfg)`

### 注入逻辑建议

1. 先找出 `adapter_model.adapter` 对应的层序列
2. 取最后 `kg_cfg.num_insert_layers` 层
3. 将每一层替换为 `KGAdapterAugmentedLayer`
4. 如果开启 LoRA
   - 仅对 KG-adapter 内部的 `q_proj/k_proj/v_proj/o_proj/up_proj/down_proj/gate_proj` 等目标层套 LoRA
5. 最后重新设置：
   - emulator 全冻结
   - base layer 全冻结
   - 仅 KG LoRA 参与训练

示意伪代码：

```python
def inject_kg_adapter_into_final_layers(adapter_model, kg_cfg):
    adapter_layers = list(adapter_model.adapter)
    n = kg_cfg.num_insert_layers
    start = max(0, len(adapter_layers) - n)

    for i in range(start, len(adapter_layers)):
        base_layer = adapter_layers[i]
        kg_module = build_kg_adapter_for_layer(base_layer, kg_cfg)
        adapter_layers[i] = KGAdapterAugmentedLayer(
            base_layer=base_layer,
            kg_adapter=kg_module,
            fusion_mode=kg_cfg.fusion_mode
        )

    adapter_model.adapter = nn.ModuleList(adapter_layers)
    adapter_model.mark_only_kg_lora_trainable()
    return adapter_model
```

## 4.5 扩展 trainer，让 batch 能接收 KG 输入

修改文件：

- `federatedscope/llm/trainer/trainer.py`

当前 trainer 默认只喂：

- `input_ids`
- `labels`
- `attention_mask`

但 KG-adapter 一般还需要额外输入，例如：

- `entity_ids`
- `relation_ids`
- `kg_edge_index`
- `node_mask`
- `subgraph_token_spans`

建议扩展 `_hook_on_batch_forward()`：

```python
kg_kwargs = {}
for key in ["entity_ids", "relation_ids", "kg_edge_index", "node_mask"]:
    if key in ctx.data_batch:
        kg_kwargs[key] = ctx.data_batch[key].to(ctx.device)

outputs = ctx.model(
    input_ids=input_ids,
    labels=labels,
    attention_mask=attention_mask,
    **kg_kwargs
)
```

同时需要让 `KGAdapterAugmentedLayer` 在前向中从 `kwargs` 里读这些字段。

## 4.6 扩展 dataloader，产出 KG 监督输入

修改文件：

- `federatedscope/llm/dataloader/dataloader.py`
- 视你的数据组织方式，可能还要新增：
  - `federatedscope/llm/dataloader/kg_adapter_dataset.py`

建议新增能力：

- 文本样本到 KG 子图的检索/映射
- 生成每个样本对应的：
  - 实体 ID
  - 关系 ID
  - 邻接边
  - token 到实体的对齐信息

如果另一篇论文的 KG-adapter 代码已经自带数据预处理，推荐只做一个适配层，把它输出的数据字段对齐到 FedBiOT 的 batch 格式。

## 4.7 聚合器保持最小改动

主要文件：

- `federatedscope/llm/llm_local/aggregator.py`

当前 `MultiLoRAAvgAggregator` 按 key 做加权平均，已经足够支持：

- `kg_adapter.*.lora_A`
- `kg_adapter.*.lora_B`

因此聚合器本身不一定要重写。

建议只加 2 个小增强：

### A. 联邦参数白名单

在 `aggregate()` 前做一次 key 过滤，只聚合：

- `llm.kg_adapter.federate_param_keywords` 命中的参数

这样更安全，防止后续有其他 trainable 参数被误上传。

### B. 可选分组聚合

如果不同客户端使用不同知识图谱域，建议后续支持：

- 全局共享 KG LoRA
- 按 domain/client cluster 分组聚合 KG LoRA

第一阶段先不做，直接复用现有 FedAvg 即可。

## 4.8 服务端/客户端流程基本不用重写

涉及文件：

- `federatedscope/llm/offsite_tuning/server.py`
- `federatedscope/llm/offsite_tuning/client.py`
- `federatedscope/core/trainers/torch_trainer.py`

原因：

当前链路已经是：

1. 服务端广播 emulator+adapter
2. 客户端本地训练
3. `trainer.get_model_para()` 返回 trainable state_dict
4. 服务端聚合并下发

只要你把“trainable state_dict”严格限定成 KG LoRA 参数，这条链就天然等价于：

- 服务端/客户端高效传输 KG-adapter LoRA

也就是说，本次重点不是改 client/server 协议，而是改：

- 模块注入
- 参数命名
- `requires_grad`
- `state_dict` 过滤

## 5. 建议新增的配置项

修改文件：

- `federatedscope/core/configs/cfg_llm.py`

建议新增：

```python
cfg.llm.kg_adapter = CN()
cfg.llm.kg_adapter.use = False
cfg.llm.kg_adapter.path = ""
cfg.llm.kg_adapter.mode = "late_fusion"
cfg.llm.kg_adapter.num_insert_layers = 2
cfg.llm.kg_adapter.fusion_mode = "residual"
cfg.llm.kg_adapter.hidden_dim = 4096
cfg.llm.kg_adapter.kg_dim = 512
cfg.llm.kg_adapter.num_heads = 8
cfg.llm.kg_adapter.dropout = 0.1

cfg.llm.kg_adapter.lora = CN()
cfg.llm.kg_adapter.lora.use = True
cfg.llm.kg_adapter.lora.r = 8
cfg.llm.kg_adapter.lora.alpha = 16
cfg.llm.kg_adapter.lora.dropout = 0.05
cfg.llm.kg_adapter.lora.target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj"
]

cfg.llm.kg_adapter.federate_param_keywords = [
    "kg_adapter",
    "lora_A",
    "lora_B",
    "kg_gate",
]
```

然后新增一套 YAML，例如：

- `federatedscope/llm/baseline/fedbiot/dolly/kg_adpt2_dp2.yaml`

核心区别是：

- `llm.offsite_tuning.use: True`
- `llm.kg_adapter.use: True`
- `llm.kg_adapter.num_insert_layers: 2`
- `llm.kg_adapter.lora.use: True`

## 6. 推荐实现顺序

为了避免一次改太多，建议按 4 个阶段推进。

### 阶段 A：只跑通结构注入

目标：

- 不接真实 KG 数据
- 先用 dummy tensor 跑通 `KGAdapterAugmentedLayer`
- 确认模型前向不报错

修改：

- 新增 `kg_adapter/`
- 在 `offsite_tuning/utils.py` 中插层

### 阶段 B：只训练 KG LoRA

目标：

- 确保上传到 server 的参数只剩 KG LoRA
- 检查 checkpoint key 是否正确

验证方式：

- 打印 `trainer.get_model_para().keys()`
- 确认只含 `kg_adapter...lora_` 类 key

### 阶段 C：接真实 KG 数据

目标：

- dataloader 输出 `entity_ids/relation_ids/...`
- trainer 能把这些字段喂给模型

### 阶段 D：做联邦策略优化

目标：

- 视需要加 domain-wise aggregation
- 加 gate/norm 的选择性同步
- 加 server-side KG prior regularization

## 7. 最关键的风险点

## 7.1 `adapter` 不是单独模块，而是一组层引用

当前 `offsite_tuning/utils.py` 里：

- `adapter_model.adapter = layers[:l] + layers[r:]`

它更像“层列表引用”而不是真正独立模块。

这意味着你不能只改 `adapter_model.adapter` 这个属性，还必须同步回：

- `adapter_model.layers`
- `set_layers(...)`

否则训练图和保存/加载图可能不一致。

## 7.2 不能把 KG 全量参数直接上传

如果 KG-adapter 内含：

- 大型 entity embedding
- relation embedding
- graph encoder projection

直接联邦同步会让通信量暴涨，也会破坏你想要的 LoRA 高效传输目标。

所以第一版必须坚持：

- 全量 KG 参数冻结
- 只同步 LoRA 和少量 gate

## 7.3 batch 结构改动会影响所有评测

一旦 trainer 强依赖 KG 字段，普通评测集也会报错。

建议让 KG 输入是“可选字段”：

- 有则用
- 没有则跳过 KG 分支或用空图占位

## 7.4 checkpoint 恢复要兼容旧模型

当前很多地方默认：

- `load_state_dict(strict=False)`

这是好事。

但你仍然要保证：

- 不开 `llm.kg_adapter.use` 时，旧 YAML 可照常运行
- 新旧 checkpoint 都能至少 `strict=False` 恢复

## 8. 建议的最小改动文件清单

第一阶段最值得改的文件只有这些：

- `federatedscope/core/configs/cfg_llm.py`
- `federatedscope/llm/model/adapter_builder.py`
- `federatedscope/llm/offsite_tuning/utils.py`
- `federatedscope/llm/trainer/trainer.py`
- `federatedscope/llm/dataloader/dataloader.py`
- `federatedscope/llm/kg_adapter/modules.py`
- `federatedscope/llm/kg_adapter/layer.py`
- `federatedscope/llm/baseline/fedbiot/dolly/kg_adpt2_dp2.yaml`

其中真正的核心代码点只有 3 个：

1. 在 `generate_emulator_and_adapter()` 后注入 KG-adapter
2. 在 `AdapterModel.state_dict()` 中只返回 KG LoRA 参数
3. 在 `LLMTrainer._hook_on_batch_forward()` 中接入 KG batch 字段

## 9. 一句话落地策略

把 KG-adapter 视为 `offsite_tuning` 产生的 adapter 区域中的“最后 K 层增强器”，并将其内部全部可训练参数收敛到 LoRA 子模块上；这样既保持 FedBiOT 现有 emulator/adapter 联邦框架不变，又能把服务端和客户端之间的通信成本严格压缩到 KG-adapter 的低秩增量参数。

## 10. 下一步建议

如果继续往下做实现，建议按下面顺序提交代码：

1. 先补 `cfg_llm.py` 与 `kg_adapter/` 模块骨架
2. 再改 `offsite_tuning/utils.py` 做插层
3. 再改 `adapter_builder.py` 做参数过滤
4. 最后改 dataloader/trainer 接入真实 KG 输入

这样每一步都可以单独验证，不容易把现有 FedBiOT 主链路一起搞坏。
