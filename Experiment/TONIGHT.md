# 今晚先跑通，再判断是否有效

先读这份；数学配套在 [GPTAQ / ResComp 推导](GPTAQ_ResComp_derivation.md) 和 [完整方案](ResComp_task_weighting_plan.md)。

## 你不必先自己训练一个大模型

用 nanoGPT 学会了 nn.Module、forward、loss，就已经有调用预训练模型的基础。from_pretrained 会加载现成架构和训练好的权重。量化实验主要改变模块权重的表示与补偿方式；不需要重新预训练。

第一步学会 tokenizer(text) → model(input_ids) → logits / CE；第二步用 get_submodule 找一个 Linear；第三步再看其 hook 和量化。架构复现可以同时慢慢学，不必作为做实验的前置条件。

本次建议从 [Qwen2.5-0.5B 基础模型](https://huggingface.co/Qwen/Qwen2.5-0.5B) 和 [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) 开始。这里评估 next-token loss，用基础模型即可，不使用聊天模板，也不以聊天表现判断预训练模型质量。

## 当前环境实测

2026-09-21 找到的解释器：

```text
C:\Users\zhuangzhe\AppData\Local\Python\pythoncore-3.14-64\python.exe
torch 2.13.0+cpu
transformers 5.15.0
torch.cuda.is_available() == False
默认 Hugging Face 缓存中未找到模型
```

这说明这个 Python 的 PyTorch 不能使用 GPU，不说明你的 4080 硬件不存在。不要用这个环境直接跑大模型并误以为在使用 GPU。

如果你已有另一个 GPU 环境，在那个环境执行下面命令即可。不替换当前全局 Python。若需要新建环境，建议单独建立 Python 3.11/3.12 环境，按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择 Windows/Pip/Python/CUDA 对应命令；安装后必须验证 CUDA。

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m pip install transformers datasets safetensors
```

本次只验证了已有 CPU 环境，没有替你安装 CUDA 包或下载预训练权重。依赖版本写入每次结果配置；CUDA 环境的版本可能不同，先跑 smoke。

## 1. 现在就能跑：不下载模型的验证

在 PowerShell 中：

```powershell
Set-Location C:\GPTQ\Experiment
$pilotPython = 'C:\Users\zhuangzhe\AppData\Local\Python\pythoncore-3.14-64\python.exe'
& $pilotPython -m unittest test_quant_core -v
& $pilotPython run_pilot.py --smoke --samples 2 --eval-samples 2 --seq-len 16 --blocksize 8 --out results/my_smoke_01
```

每次用新的 --out，避免覆盖结果。首次导入 torch/transformers 在本机可能较慢。smoke 用随机小 Llama 和随机 token，只证明数据流可运行，不能作为量化质量结论。

## 2. 学模型调用：只读一个短脚本

读 [learn_load_model.py](learn_load_model.py)，然后在 CUDA 环境运行：

```powershell
python learn_load_model.py --device cuda --allow-download
```

这一步首次会下载 Qwen2.5-0.5B。你应看到目标权重形状、输入形状、logits 形状和一段续写。后续可去掉 --allow-download 使用缓存。若有本地模型目录，使用 --model '你的目录'。

模型名不是你的模型类；AutoModelForCausalLM 会根据 config 选用实现，from_pretrained 再载入权重。model(**inputs) 做一次前向，model.generate(**inputs) 才是迭代生成。

## 3. 准备真实文本

```powershell
python prepare_data.py --out data/wikitext2_seed0 --seed 0
```

这个命令需要 datasets，并下载 WikiText-2。脚本将 train 与 validation 分别写为 calibration.jsonl、validation.jsonl，去除完全重复文本，记录来源行与文件哈希；不读取 test split。

也可以自己提供两个 UTF-8 JSONL 文件，每行形如：

```json
{"text": "A sufficiently long document..."}
```

校准、验证必须独立。脚本在每段文本内切出无 padding 的固定长度窗口，不跨文档拼接；短文本跳过，数量不够会报错，不会静默重复数据。它是 pilot 协议，不等于标准 WikiText 整体 PPL 评测协议。

## 4. 今晚第一个真实对照

```powershell
python run_pilot.py --model Qwen/Qwen2.5-0.5B --allow-download --device cuda --dtype bfloat16 --calibration data/wikitext2_seed0/calibration.jsonl --validation data/wikitext2_seed0/validation.jsonl --target model.layers.12.self_attn.o_proj --samples 8 --eval-samples 8 --seq-len 128 --bits 4 --groups 4 --rho 0.5 --out results/qwen05b_pilot_s0
```

先小规模跑通。若显存和时间允许，再将 samples/eval-samples 改为 32、seq-len 改为 512；再跑 ρ=0.25、0.5、1 和额外 seed。不要今晚一开始就同时做很多层和很多超参数。

若 CUDA 报错，脚本不会偷偷退回 CPU。如果 BF16 环境不支持，选 float16 后先检查梯度分位数是否大量归零。

## 5. 实验实际比较了什么

程序执行顺序：

1. 加载一个预训练模型，计算浮点验证 CE。
2. 在校准文本上采集浮点目标模块输入、任务输出梯度。
3. 将目标所在 block 之前的完整 blocks 用 RTN 量化，所有候选共享这一个前缀；当前 block 的其他模块保持浮点。
4. 采集这个前缀产生的 X_q，保持教师 X_f 与原始 W₀固定。
5. 比较 RTN 目标模块、GPTAQ 分解参考、ResComp 官方模块核心、加权官方核心、打乱权重对照。
6. 在同一验证文本上计算 CE/PPL，保存每序列 CE、重建误差和候选模块权重。

没有训练模型，也没有逐候选重新改变前缀。只保存目标模块的 fake-quant 浮点权重，不生成压缩推理 checkpoint。

**报告名称应为“共同 RTN 前缀下的单模块 pilot”。** 使用原官方 fasterquant 核心不等于复现其官方完整模型管线。这里采用固定逐输出行对称网格，关闭旋转、actorder 和输入 group scale；所有候选一致。GPTAQ 标签带 reference，表示自己的可读实现。

这是为了今晚先判断任务敏感度是否有用；后续必须替换为完整的原版 ResComp 前缀和正式评估协议。

## 6. 到哪里看结果

* config.json：模型、版本、参数、精确 token 窗口哈希。
* diagnostics.json：输入漂移、梯度分位数、公共前缀模块数。
* metrics.json：所有对照的 CE/PPL、calibration MSE、task MSE、validation MSE、时间与显存。
* 各 method.pt：仅目标模块的权重和名称，可用来比较量化差异。

看 task_weighted_rescomp.delta_ce_vs_rescomp：负数表示这一次验证 CE 更低。不要只看校准加权误差；也不要把微小单次差值称为已经证明有效。当前输出是验证结果，没有独立最终测试结果。

基线 quantization_seconds 包含其内核运行；加权方法包含分组调用。总时间还包括模型加载、梯度采集与评测，应一起记录；这些不是正式性能 benchmark。

## 7. 哪些代码留给你深入修改

骨架提供可运行默认实现，不以 TODO 阻止今晚运行。建议你首先重写 quant_core.py 中标注 USER EDIT POINT 的 make_weights 和 statistics，用自己的实现通过测试；这是你的研究改动核心。

随后自己补充分组策略、不同敏感度代理、结果汇总与完整逐层管线。保留原版和打乱对照，不在得到不理想结果时直接删去。

官方文件固定到 commit 354a96255e1738e4063212599402503e43e2013d，原文件与 SHA256 在 vendor/rescomp。适配器只执行已核查类，不导入官方整个训练工程。若源码丢失，可运行 python fetch_rescomp_core.py 重新下载；不自动执行远程更新。

## 8. 今晚你能向导师汇报的内容

如果只跑完 smoke：报告“公式与数据流验证通过，真实模型实验待完成”。

如果完成预训练模型 pilot：报告模型/位宽/公共前缀、单个目标模块、校准和验证 token 数、基线与加权的 CE 差、随机权重对照和额外成本。即使结果为负，也能据梯度分布、输入漂移与误差指标定位下一步。

不要报告“已经全面超过 ResComp”，除非完整模型、公平超参预算及独立测试支持这个结论。
