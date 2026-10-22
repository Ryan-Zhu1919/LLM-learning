# LLM-learning

这是一个围绕大语言模型原理、代码实现和工程实践持续建设的个人学习项目。通过从零实现 Transformer 相关模块、探索真实开源大模型的架构与推理流程，并逐步开展微调实践，建立从模型内部计算到实际使用的系统理解。

项目以 Python 和 PyTorch 实践为主，用 Notebook 记录代码、张量形状、解释和实验输出，便于复现、复习与继续扩展。目前已完成简化 MiniGPT 的基础实现与小规模训练、生成练习，正在通过 Qwen 探索真实模型；监督微调仍处于计划阶段。这里的实现面向学习，不是完整的大模型训练框架，也不代表工业级训练或部署能力。

## Learning Roadmap / 学习路线

### 阶段一：Transformer 基础与 MiniGPT 从零实现 · 基础练习已完成

目录：[miniGPT-from-scratch/](miniGPT-from-scratch/)

从张量运算、`nn.Module` 和训练循环出发，手写缩放点积 Attention，再将 Token Embedding、正弦位置编码、多头因果注意力、FFN、残差连接和 LayerNorm 组合为 Decoder-only MiniGPT，串起前向计算、Next Token Prediction 训练与自回归生成。

这个阶段的目标是通过自己编写代码理解模型如何运行。现有实验使用两条合成整数序列训练 20 个 epoch，保存了训练 loss 和生成 Token ID 的输出；它没有文本 Tokenizer、语料训练或验证集，不能据此评价自然语言能力。MiniGPT 使用 Post-LayerNorm、ReLU FFN 和正弦位置编码，是学习用的简化实现，与 MiniGPT-4 无关。

### 阶段二：真实大模型架构与推理探索 · In Progress / 探索中

目录：[llm-architecture-exploration/](llm-architecture-exploration/)

使用 Transformers 加载 `Qwen/Qwen2.5-0.5B` 基础模型及配套 Tokenizer，将阶段一的概念对应到真实模型：

- 沿文本编码、Embedding、隐藏状态、`lm_head` 和 Logits 追踪数据与 shape。
- 阅读 Decoder Layer，观察 RMSNorm、门控 MLP 与 MiniGPT 的差异；手动展开第一层 Attention 中的 Q/K/V 投影、GQA、RoPE 和因果遮罩。
- 使用 `model.generate()`，并编写逐 Token 贪心生成循环，对照未复用历史缓存与传递 `past_key_values` 的过程。
- 对生成不同数量 Token 的端到端耗时进行预热、重复测量并取中位数，观察平均 Token/s。

这是基于预训练模型的结构探索与推理实践，不是从零复现完整 Qwen。现有保存输出展示了前向、生成和 CPU 计时结果；手动 Attention 尚未与模型内部结果进行数值一致性比较，计时也没有验证 KV Cache 的加速倍数。

### 阶段三：监督微调与参数高效微调 · Planned / 计划开展

尚未正式完成，当前没有独立的微调模块。计划从以下问题入手：

- 训练数据如何组织为 `input_ids`、`labels`，Causal Language Modeling 的训练目标与 Loss 如何对应。
- SFT 的数据准备、训练与评估流程。
- LoRA、QLoRA 的基本原理与实践。
- 微调模型或适配器的训练、保存、加载及效果评估。

## Repository Structure / 项目结构

```text
LLM-learning/
├── README.md                              # 仓库总览、学习路线与环境说明
├── requirements.txt                       # 当前代码及 Notebook 工具的公共依赖
├── miniGPT-from-scratch/
│   ├── README.md                          # MiniGPT 实现细节与实验边界
│   ├── requirements.txt                   # MiniGPT 独立运行依赖
│   └── notebooks/
│       ├── 01_transformer_foundation.ipynb # PyTorch、训练循环与 Attention 基础
│       └── 02_miniGPT_from_scratch.ipynb   # MiniGPT 组装、训练与生成
├── llm-architecture-exploration/
│   ├── forward_pass.ipynb                 # Qwen 单次前向与下一 Token 预测
│   └── load_qwen.ipynb                    # Qwen 结构、生成、缓存与计时
└── Draft Code/                            # 草稿、早期实现与复习练习
    ├── chapter5.ipynb
    ├── transformer.ipynb
    ├── review.ipynb
    ├── forward_pass.ipynb
    └── load_qwen.ipynb
```

根目录 README 负责整个仓库的学习脉络与入口；[MiniGPT 子项目 README](miniGPT-from-scratch/README.md) 继续说明该子项目的组件、配置、运行方式和实验范围。

## Notebook Guide / Notebook 导航

推荐主线：**Transformer Foundation → MiniGPT From Scratch → Qwen 前向传播 → Qwen 结构与推理探索**。每份主要 Notebook 都包含自己的导入与初始化，阅读顺序不意味着需要共享运行时变量。

### 1. [Transformer Foundation](miniGPT-from-scratch/notebooks/01_transformer_foundation.ipynb)

- **学习目标**：读懂 PyTorch 张量、模型封装与梯度更新，为理解 Transformer 计算打基础。
- **核心内容**：矩阵运算与 `(B, T, D)` shape；`nn.Linear`、`nn.Module` 和参数；求和回归任务中的 MSE、反向传播与 Adam；手写缩放点积 Attention 和 Token Embedding。
- **建议阅读顺序**：作为第一份 Notebook，完成后进入 MiniGPT，将独立组件连接起来。

### 2. [MiniGPT From Scratch](miniGPT-from-scratch/notebooks/02_miniGPT_from_scratch.ipynb)

- **学习目标**：从组件组装一个可训练、可生成的简化 Decoder-only Transformer。
- **核心内容**：Embedding 与正弦位置编码；多头因果注意力、FFN、残差与 Post-LayerNorm；Decoder 堆叠和 LM Head；shifted targets、交叉熵训练与温度采样生成。
- **建议阅读顺序**：接在 Foundation 之后，沿 `forward()` 追踪输入到 Logits，再阅读 `generate()`；随后转向真实模型的前向传播。

### 3. [Qwen 前向传播：从文本到下一个 Token](llm-architecture-exploration/forward_pass.ipynb)

- **学习目标**：理解真实预训练模型如何把文本变成下一 Token 的预测。
- **核心内容**：加载模型与 Tokenizer；`input_ids`、`attention_mask` 和 Embedding；完整前向的 Logits 及配置维度核对；取最后位置分数并用 `argmax` 预测一个 Token。
- **建议阅读顺序**：在 MiniGPT 之后阅读，先建立真实模型的数据流概念，再进入 `load_qwen.ipynb`。本份只做单步预测，没有连续生成循环。

### 4. [Qwen 结构与推理探索：从加载到 KV Cache](llm-architecture-exploration/load_qwen.ipynb)

- **学习目标**：把真实模型的内部模块、生成流程与推理开销联系起来。
- **核心内容**：模型对象树、Decoder Layer 与 GQA 投影维度；含 RoPE 的第一层 Attention 手动计算；最终隐藏状态、Logits、自动与手动生成及 KV Cache；固定新增 Token 数的端到端计时。
- **建议阅读顺序**：接在 `forward_pass.ipynb` 之后。先观察结构与 shape，再比较生成循环中输入长度和缓存传递方式，最后阅读计时范围与限制。

### 补充：Draft Code / 草稿与复习

这部分保留探索过程，建议完成主线后按需查阅，不作为统一的顺序运行入口：

- [chapter5.ipynb](Draft%20Code/chapter5.ipynb)：张量、模型封装、训练循环、Attention 和位置编码练习；含直接查询 CUDA 设备的单元格，CPU 环境需跳过该查询。
- [transformer.ipynb](Draft%20Code/transformer.ipynb)：Transformer 编码器分类与早期 MiniGPT 草稿；生成函数在循环内返回，阅读完整生成流程请使用主线 MiniGPT Notebook。
- [review.ipynb](Draft%20Code/review.ipynb)：Embedding、因果及多头注意力、TinyGPT 训练与前向拆解复习；包含 `torch.allclose` 比较代码，但部分单元格没有执行记录，不能视为已完成验证。
- [forward_pass.ipynb](Draft%20Code/forward_pass.ipynb) 与 [load_qwen.ipynb](Draft%20Code/load_qwen.ipynb)：Qwen 前向与推理的代码草稿，优先阅读上方带完整讲解的对应 Notebook。

## Environment Setup / 环境配置

### 创建独立环境并安装依赖

在终端进入仓库根目录 `LLM-learning/`。以下以 Windows PowerShell 为例，通过环境内的 Python 直接执行命令，无需先激活环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

macOS / Linux 可使用对应路径：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

选择与拟安装的 PyTorch、Transformers 版本兼容的 Python。现有 Notebook 的 Python 元数据记录为 3.9.25，这不是本仓库对当前依赖版本组合的兼容性保证。

### 在 VS Code 中运行

1. 用 VS Code 打开整个 `LLM-learning/` 文件夹，安装 Python 和 Jupyter 扩展。
2. 通过 **Python: Select Interpreter** 选择新建的 `.venv`，打开主线 Notebook，再在右上角选择该环境作为 Kernel。
3. 每份 Notebook 从头按顺序执行；需要复现时先重启 Kernel，避免旧变量影响结果。先运行 Foundation 与 MiniGPT，再尝试加载 Qwen。

### 依赖关系与运行边界

根目录 [requirements.txt](requirements.txt) 覆盖当前 Notebook 的直接第三方计算依赖与运行工具：

| 依赖 | 代码来源 / 用途 |
|---|---|
| `torch` | 所有 Notebook 的张量、神经网络、自动求导、训练与推理；沿用 MiniGPT 依赖 |
| `transformers` | 两个目录中的 Qwen Notebook 使用 `AutoTokenizer`、`AutoModelForCausalLM` 和生成接口 |
| `jupyterlab` | 浏览器中的 Notebook 阅读与执行；沿用 MiniGPT 依赖 |
| `ipykernel` | Python Notebook 内核，供 VS Code / Jupyter 选择环境 |

`math`、`time`、`statistics` 属于 Python 标准库。当前代码没有直接使用 NumPy 或 IPython API，也没有使用 PEFT、TRL 等微调库，因此不额外列为直接依赖。现有子项目依赖没有版本约束，根目录同样没有加入未经验证的版本范围；它是依赖清单，不是锁定环境。

若只运行 MiniGPT，可在另一个独立环境中、从仓库根目录安装其原有依赖：`python -m pip install -r miniGPT-from-scratch/requirements.txt`，然后使用 `python -m jupyterlab`。该文件仅包含 `torch`、`jupyterlab`，不覆盖 Qwen 所需的 `transformers`；已有可用 MiniGPT 环境可以继续保留，Qwen 探索使用单独环境。

MiniGPT 主线示例默认使用 CPU，无需下载数据或模型。Qwen Notebook 首次调用 `from_pretrained()` 时可能需要联网下载 Tokenizer、配置与权重，后续可以使用本地缓存；只有缓存资源完整时才具备离线运行的条件。现有加载代码没有自动将模型迁移到 GPU，输入按模型所在设备对齐。

模型能否运行取决于设备、可用内存与显存、精度、模型规模及具体依赖版本。尤其是 `model.model.layers`、`rotary_emb` 等内部接口和 KV Cache API 可能随 Transformers 版本变化。**本次环境与依赖说明基于静态代码检查，未重新安装依赖或执行 Notebook，不能据此认为全部 Notebook 已在同一环境测试通过。** 保存输出只记录此前运行时的结果。

## Project Status / 当前进度

| 状态 | 内容与证据 |
|---|---|
| 已完成的基础学习与代码实践 | 两份 MiniGPT 主线 Notebook 包含 PyTorch 基础、手写 Attention、简化模型、合成序列训练和生成，留有对应输出；草稿另有编码器分类与 TinyGPT 复习代码 |
| 已开展、仍在探索 | Qwen 前向、结构、GQA / RoPE、逐 Token 生成、KV Cache 及基本 CPU 计时已有代码与保存输出；尚无完整模型复现、手动 Attention 数值对齐或缓存加速对照结论 |
| Planned / 计划开展 | 真实模型 SFT、LoRA / QLoRA、训练数据处理、模型或适配器保存加载及效果评估，当前没有对应训练成果 |

## Future Work / 后续计划

- 从小规模真实模型的 SFT 与 LoRA 实践入手，理解训练目标与参数更新方式。
- 补充数据处理、训练与评估流程，区分训练 loss、生成样例和实际效果。
- 根据学习进展与算力条件，逐步探索其他后训练方法或 Agent 方向。
