# Guided residual reconstruction：学习与实验起点

这是待验证的研究假设，不是已经成立的新算法或已完成复现。
目标：把端任务损失的输出敏感度引入 ResComp 的固定 FP 目标重构。
核心算法由你实现；先验证数学与退化条件，再接大模型。

## 已知硬件与方向澄清

用户设备：RTX 4080 Laptop，12 GB 显存。系统执行环境和现有模型待确认。
先 CPU toy，再小模型单模块，最后全模型。7B 半精度权重约 14 GB（十进制）本身已超 12 GB，不能默认整模型加梯度放得下。
真实模型大小要结合官方实现支持、可取得的权重与 measured peak memory 决定；暂不下载或安装。
用户补充并未指明另一篇 GPTAQ 加权论文，而是在提出梯度加权设想，不能把它记成已确认的先行工作。
直接对权重误差按权重梯度平方加权，和对输出误差按输出梯度平方加权，是两个不同目标。

## 当前核查（2026-09-21）

- Experiment/model.py 已存在且为空，本次不修改。
- 本机 ResComp/code、GuidedQuant/code 当前为空；阅读指南是旧快照，不能作为源码已就绪的证据。
- MARR/code/README.md 只是占位说明；本次检索未确认可用官方实现。
- 当前终端 PATH 找不到 python，未运行本练习。先使用你实际配置的 Python 解释器。
- 尚未进行完整 novelty 检索；用户提到的 GPTAQ 加权论文还需要标题或链接。

## 第一份作业（只用 Python 标准库）

1. 阅读 math_notes.md 的“固定通道权重为什么可能无效”。
2. 实现 exercise_01.py 的 weighted_error 和 best_candidate 两个函数。
3. 执行：python -m unittest -v test_exercise_01.py
4. 解释：为何统一乘一个正数不改变最优候选，而对不同 token 加不同权重会改变？
5. 把实现和解释交给助手审阅；先不要写 Transformer 或完整量化器。

测试目前应因 NotImplementedError 失败，这是有意保留的练习状态，不是可运行量化实现。

## 后续里程碑与验收

1. 小型加权最小二乘：实现 H、b，用有限差分验证梯度；全 1 权重必须退化为普通重构。
2. 固定一个权重为量化值：自己推导带等式约束的更新；与直接约束最小二乘对照。
3. CPU 小矩阵量化：实现无 block 优化的逐列参考算法。记录原始 W0、当前 W、量化 Q，固定目标 Y0；明确已量化列贡献，不能重复计算残差。
4. 源码复现：获取官方 GPTAQ / ResComp / GuidedQuant 仓库并记录 commit、依赖、运行命令；先查缺失导入，不能直接拼不同仓库然后称为官方基线。
5. 真实模型单模块：抽取同一批 token 对应的 X_fp、X_quant、Y_fp、输出梯度 G。首先只动一个线性投影模块，固定其他模块。
6. 分组版本：把输出通道分组，共享组内 token 权重，分别收集 H 与残差交叉统计；先核查显存再增加组数。
7. 全模型：统一模型、校准数据、量化网格、位宽、group size、列顺序、阻尼、旋转、残差系数。测试集不得参与选择敏感度和超参数。

## 必需对照

- FP、RTN、GPTQ、GPTAQ、ResComp。
- ResComp + 固定通道正权重（理论不变性对照，按行可分、网格固定）。
- ResComp + token/通道梯度平方权重（核心候选）。
- ResComp + 打乱 token 对应的敏感度（验证收益是否来自任务对齐）。
- 固定残差系数 sweep：先排除收益仅来自调 alpha。
- MARR 若需自实现，标成 unofficial reproduction，独立验证后再作为结论性对照。
- 最后才考虑把 residual scaling 与 sensitivity weighting 组合，不同时改多项。

记录：未加权 MSE、加权误差、held-out PPL、量化码是否改变、耗时、峰值显存、失败/非有限值、多个校准 seed。
MSE 下降不保证 PPL 下降；toy 测试成功不表示 ResComp 复现成功。

## 你写什么，助手做什么

你：目标函数、统计量收集、约束更新、逐列循环、实验解释。
助手：拆任务、推导与维度检查、提供独立验收、审查 diff、定位错误、协助环境与实验记录。
默认不替你一次写完核心算法。

## 阅读顺序（按问题读）

GPTQ：单行二次型与补偿更新 → GPTAQ：FP/quant 两条输入与残差 → ResComp：式 12–21、Algorithm 1 的 W0/W(q) 区别 → GuidedQuant：式 4、6、7 与分组近似 → MARR：残差缩放及模块级系数。

官方资料：
- https://arxiv.org/abs/2504.02692
- https://arxiv.org/html/2604.07955v1
- https://github.com/list0830/ResComp
- https://arxiv.org/html/2505.07004v4
- https://github.com/snu-mllab/GuidedQuant
- https://arxiv.org/html/2605.17997v1

目前在线 ResComp 核心入口：fake_quant/gptaq_utils_r.py 中 add_batch、fasterquant。
它不仅用 H，还用 dXXT 和由二者组合出的交叉统计。只改 H 不足以实现一致的加权残差目标。
该实现还有 alpha、alpha2 和按位宽切换的 mode；复现时需要记录，不能默认等于论文的理想公式。
