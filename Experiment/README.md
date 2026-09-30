# 任务加权 ResComp 实验区

更新时间：2026-09-28。这里保存学习材料、可运行代码和真实实验结果。

**资源保护：** 第五至七轮使用4 GiB显存分配预算、CPU缓存及显存/温度检查。第七轮额外采用150ms间歇和≤65℃启动条件，硬停止仍为78℃。运行前看 [安全运行说明](RUN_SAFELY.md)；第一至四轮历史脚本不会自动继承这些保护。

## 先读哪几个文件

1. **先理解想法**：[用白话读懂实验](docs/START_HERE.md)。解释我们改了什么、每个指标是什么意思。
   新增：[详细白话技术说明](docs/TECHNICAL_WALKTHROUGH.md)，逐步解释七轮改动、公式、代码、选参、结果及资源保护。
2. **看第一轮结果**：[第一轮实验报告](reports/20260926_report.md)。真实 Qwen 模型，3 个校准 seed，收益很小但可以运行。
3. **看最新结论**：[完整实验报告](reports/final_report.md)。按轮回顾方法、成功与负结果、外部语料验证、资源保护及是否继续研究的判断。
4. **再读实现**：先看 `quant_core.py` 的 `make_weights` 和 `statistics`，再看 `run_pilot.py` 的 `main`。

**第七轮：[保守修复与双域选参](reports/round7_report.md)**。比较量化码收缩、块输出均值补偿及双域选参，使用新的PTB validation与Shakespeare窗口确认。第六轮的PTB test已转为开发数据，不再称为未见测试。设计与修订见 [第七轮协议](reports/round7_protocol.md)。原梯度加权尚未证明稳定超过强基线。

当前结果：冻结的组合方案在2层×2新语料均平均改善初始GPTQ，且各3/3seed；但第18层简单均值偏置更好，组合方案在该层Shakespeare的区间仍跨0。建议有限继续验证组合泛化，不扩大梯度权重搜索。

历史报告：[第一轮](reports/20260926_report.md)、[第二轮](reports/20260926_round2_report.md)、[第三轮](reports/round3_report.md)、[第四轮](reports/round4_report.md)、[前五轮完整报告](reports/round5_final_report.md)、[第六轮](reports/round6_report.md)。历史结论反映当时证据，当前判断以完整报告最前面的第七轮为准。

## 文件放在哪里

| 位置 | 内容 |
|---|---|
| docs/ | 白话说明、学习路线、数学推导、相关工作 |
| reports/ | 人看的实验报告和事先确定的实验方案 |
| learning/ | 小练习、模型加载示例、原来的空白 model.py |
| archive/ | 过时的阶段安排和历史环境记录 |
| results/ | 机器生成的指标、配置、目标模块权重 |
| data/ | 校准和验证文本；不是模型权重 |
| vendor/rescomp/ | 固定版本的官方核心和来源校验 |
| .hf_cache/ | 下载的预训练模型和数据缓存，无需阅读 |

| 根目录代码 | 作用 |
|---|---|
| quant_core.py | 量化方法与加权统计量 |
| run_pilot.py | 单次实验；--refine 扩展为第二轮对照 |
| evaluate_refinement.py | 按验证集冻结配置，再进行独立测试 |
| prepare_round3_holdout.py | 按文章隔离新的评估数据，排除全部实际校准文章 |
| evaluate_iteration.py | 冻结第三轮所有层的选择，再统一评估预留文章 |
| run_round3.ps1 | 第三轮扫描与评估命令；已有结果时拒绝覆盖 |
| build_round4_prefix.py | 按校准 seed 构建块级 GPTQ 公共前缀 |
| prepare_round4_holdout.py | 排除校准文章和第三轮文章，留出第四轮新评估集 |
| evaluate_round4.py / run_round4.ps1 | 第四轮跨前缀、跨模块扫描及冻结后的评估 |
| test_round4.py / diagnose_round4_prefix_device.py | GPTQ 核心对照与 CPU/CUDA 量化格点诊断 |
| run_round5.ps1 / gpu_safety.py | 第五轮顺序执行入口与资源预算；异常停止后续队列 |
| run_round5.py / run_round5_block.py | 固定网格下的分支、合流、MLP及含残差完整块目标 |
| evaluate_round5.py | 所有配置冻结后，在新文章上评估并记录资源 |
| run_round6.py / run_round6_bounded.py | 两个目标层的敏感度、训练预算、均值偏置及受限方向对照 |
| evaluate_round6.py / prepare_round6_holdout.py | 第六轮冻结后PTB评估及数据身份 |
| run_round7.py / run_round7.ps1 | 顺序执行量化码收缩、均值修正和开发评估 |
| evaluate_round7.py / prepare_round7_holdout.py | 双域/Wiki单域选参冻结与两份新语料确认 |
| round7_safety.py | 第七轮额外间歇与启动降温条件 |
| test_round6.py / test_round7.py | 方向罚项数学、码值收缩及选择器检查 |
| prepare_data.py | 准备 WikiText 校准/验证文本 |
| test_quant_core.py | 检查核心实现 |
| test_refinement.py | 检查第二轮选参规则 |
| test_iteration.py | 检查RMS权重数值、尺度不变性和退化行为 |
| verify_weighted_math.py | 检查数学恒等式 |
| fetch_rescomp_core.py | 官方核心缺失时按固定版本下载 |

## 学习材料怎么选

| 你的问题 | 阅读文件 |
|---|---|
| 完全不懂这次实验在干什么 | [白话说明](docs/START_HERE.md) |
| 想按顺序补论文和推导 | [学习路线](docs/STUDY_ROADMAP.md) |
| GPTAQ、ResComp 的公式怎么来 | [推导参考](docs/GPTAQ_ResComp_derivation.md) |
| 我的不同方案应该怎么写成数学 | [方案工作册](docs/SCHEME_DERIVATIONS.md) |
| 哪些加权方式其实不起作用 | [数学检查点](docs/math_notes.md) |
| 想看原始完整研究设计 | [完整研究方案](docs/ResComp_task_weighting_plan.md) |
| 可能与哪些工作重叠 | [2026-09-21 检索记录](docs/RELATED_WORK_2026-09-21.md)；是当时记录，不是本次重新查新 |

`learning/exercise_01.py` 故意保留两处待实现函数，练习测试失败不代表正式实验失败。`learning/model.py` 是原来的空文件，不是实际使用的模型；实际模型通过 Transformers 加载。

## 怎么看结果目录

`20260926_qwen05b_w4_s0/s1/s2` 是第一轮真实模型实验。名字含 `smoke` 的目录只验证程序能跑，不能用于判断方法有效。第二轮结果见 [结果目录说明](results/README.md)。

| 文件 | 读它能知道什么 |
|---|---|
| metrics.json | CE/PPL、重建误差、耗时；_summary 存在表示流程完整结束 |
| config.json | 模型版本、参数、校准/验证窗口哈希、代码哈希 |
| diagnostics.json | 输入漂移、梯度分布、量化前缀大小 |
| *.pt | 单个目标模块的假量化权重，不是完整模型或压缩部署文件 |

第一轮机器汇总：[20260926_summary.json](results/20260926_summary.json)。报告是解释，JSON 是原始证据；两者都保留。

## 怎么运行

在 PowerShell 中进入本目录，使用现有 Python 和已下载缓存：

```powershell
Set-Location C:\GPTQ\Experiment
$env:HF_HOME='C:\GPTQ\Experiment\.hf_cache'
$env:HF_HUB_OFFLINE='1'
python -m unittest test_quant_core verify_weighted_math test_refinement test_iteration test_round4 test_round5 -v
```

真实实验命令在各轮报告和协议中。每次使用新的输出目录，程序会拒绝覆盖已有结果。不要运行所有练习测试来代替上面的核心测试。

`archive/` 中“今晚”“不下载”“暂不实验”的内容是历史阶段安排，当前以本入口和最新报告为准。已经运行成功，不表示你已掌握推导；你可以按学习路线逐步补上。
