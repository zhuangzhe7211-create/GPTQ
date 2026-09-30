# 相关工作核查：任务敏感度引导的残差补偿量化

检索日期：2026-09-21。结论基于本次可访问的公开论文和作者仓库，不是穷尽性查新报告；搜索未命中不等于不存在。

## 1. 先给结论

**大方向有人做，而且有高度相关工作。** 最直接的是 GuidedQuant：输出梯度平方加权、保留同一输出通道内的权重相互作用、输出通道分组，都已有明确先例。[GuidedQuant §3](https://arxiv.org/html/2505.07004v4#S3)

本次没有找到明确以“ResComp 的固定 FP 目标 + 分组 token 梯度平方 + 对 H/C/T 一致加权 + 静态解析逐列补偿”为完整方法描述的公开论文。这个表述只说明本次检索的证据边界，不能据此宣称首创。

我的判断：目前适合把课题定位为**已有任务度量与补偿机制的结合、机制验证和成本分析**。单独替换加权统计量是自然扩展；仅有组合和正向结果，未必足以构成论文贡献。先学明白再决定差异点，不急着给方法取名。

## 2. 与你的方案逐项比较

| 工作与原始来源 | 已经覆盖什么 | 与当前设想的关系 | 本次核查深度 |
|---|---|---|---|
| [GPTAQ，2504.02692v3](https://arxiv.org/html/2504.02692v3) | 非对称输入校准，残差约束更新与高效分解 | 数学基础；搜索时也搜旧名 GPTQv2 | 方法 §4、附录入口与公式核查 |
| [ResComp，2604.07955v1](https://arxiv.org/html/2604.07955v1) | 固定原始输出，加入 compensation-aware error | 你拟扩展的补偿机制 | §3–4、式 (10)–(21)、算法 1 核查 |
| [GuidedQuant，2505.07004v4](https://arxiv.org/html/2505.07004v4) | 任务输出梯度加权与输出分组 | A/B 最直接的已有技术；不能声称这些构造是新提出的 | §2–3、式 (4)–(7)、算法 1 核查 |
| [MARR，2605.17997v1](https://arxiv.org/html/2605.17997v1) | 按模块调节残差强度，以重建误差反馈估计系数 | 若 C 选补偿系数，必须对照；“系数”和“误差度量”不同 | §3 的目标与 PID 机制核查 |
| [YAQA / Model-Preserving Adaptive Rounding，2505.22988](https://arxiv.org/abs/2505.22988) | 全模型 KL 的层级 Kronecker 曲率近似及自适应舍入 | “让量化更贴近最终模型损失”已有系统工作；与 D 相关 | 摘要级核查，未逐式审计 |
| [KronQ，2607.07964](https://arxiv.org/abs/2607.07964) | 梯度协方差，用于双向旋转与混合精度分配 | 任务敏感性不只表现为误差加权；与 D 的曲率背景相关 | 摘要与作者仓库入口核查 |
| [BaKron，2608.06291](https://arxiv.org/abs/2608.06291) | 两侧 Kronecker 曲率下更高效的自适应舍入求解 | 若尝试跨输出耦合，不能忽略已有高效求解路线 | 摘要级核查 |
| [REAL-Q，2609.00049v1](https://arxiv.org/html/2609.00049v1) | 聚合 Fisher 的 block 输出目标，逐列块动态梯度修正 | 同时考虑任务敏感性和传播误差的近期近邻；不是同一个 ResComp 静态更新 | §3–4 核查 |

另有更早的 [BRECQ](https://arxiv.org/abs/2102.05426)，属于 block reconstruction 背景；GuidedQuant 的相关讨论明确关联了 Fisher 加权与早期压缩方法。它不是当前阅读主线，后续写 related work 时再补原文。

## 3. 最应改变原计划的两点

### 3.1 A/B 的加权及分组本身已有先例

按本项目列样本约定，候选形式是

$$
J(Q)=\tfrac12\sum_{i,t}s_{it}\bigl[(QX_q-W_0X_f)_{it}\bigr]^2.
$$

你要研究的差异应落在 X_f / X_q 不同、原始目标固定、补偿过程中残差的处理，以及收益与成本；不能只把普通 Gram 矩阵换成加权 Gram 就称为新理论。这个判断是对本地方案与论文的比较，不是论文替你证明了组合有效。

### 3.2 不能把“上下游一起考虑”当作未被探索的空白

REAL-Q 已明确面向任务敏感度和传播误差，并引入动态梯度修正。你的当前设想仍有不同的求解方式和成本结构，但需要据此缩小论点。该论文没有因提到 GPTAQ / GuidedQuant 就证明与你的方法完全相同。[REAL-Q 方法](https://arxiv.org/html/2609.00049v1#S4)

论文的陈述也需要核对：REAL-Q 引言把 GPTQ 描述为只使用干净输入，而 GPTAQ / ResComp 对 GPTQ 的基线说明允许使用 Quant-flow。阅读时用明确的输入/目标公式比较，不直接复制任何一篇对前作的概括。

## 4. 可以研究，但尚未证明的新问题

1. **度量与残差处理是否存在交互？** 比较无/有任务权重 × GPTAQ/ResComp 补偿，判断收益是普遍的加权效果还是与 compensation-aware 项有特别联系。
2. **FP 教师敏感度在输入漂移后是否仍可靠？** 当前先固定教师 G；量化路径重估是另一个变量，以后才考虑。
3. **额外统计是否值得？** 共享 token 权重是否已经足够，输出分组是否提供稳定收益？组数增长的成本是否抵消收益？
4. **加权代理是否更会排序候选？** 比较 J₀、J_s 与独立 CE/KL 的排序一致性；代理下降但任务变差同样是有信息的结果。

这些是研究问题建议，不是“没人做过”的结论。未来若找到完全相同的组合，仍可作为学习复现，但要更新创新性定位。

## 5. 本次检索范围与可复查路径

使用公开网页搜索，并打开 arXiv 方法原文；优先采用论文与作者仓库，不用自动生成的论文解读作为结论依据。

代表检索式：

- `ResComp quantization task gradient weighted GuidedQuant`
- `"ResComp" "gradient"`、`"ResComp" "GuidedQuant"`、`"ResComp" "weighted" quantization`
- `"GPTAQ" "GuidedQuant"`
- `quantization residual compensation task sensitivity Fisher weighted reconstruction 2026`
- `"asymmetric" "quantization" "Fisher" gradient`
- `"REAL-Q" quantization arxiv`、`"YAQA" quantization arxiv`、`"KronQ" arxiv`

作者仓库入口：[ResComp](https://github.com/list0830/ResComp)、[GuidedQuant](https://github.com/snu-mllab/GuidedQuant)、[KronQ](https://github.com/Intelligent-Computing-Lab-Panda/KronQ)。本次查新没有逐个审计仓库分支、提交、issue 或未发表实现，也没有运行这些基线。

GPTAQ 的 arXiv 编号在部分索引中仍显示旧名 GPTQv2，应同时搜两个名字。当前学习固定 v3；ResComp 固定 v1，避免公式编号随版本变化。

## 6. 不要让查新变成阅读负担

现在精读：GPTAQ → ResComp → GuidedQuant §3。MARR 与 REAL-Q 先知道差异，在完成自己的推导后再比较方法。YAQA / KronQ / BaKron 先保留在相关工作表里；只有推进非对角方案 D 时才深入。

