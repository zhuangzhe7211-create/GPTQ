# 数学合同：先定义目标，再修改更新

以下为本项目候选目标的独立推导，尚未证明实用效果。

GPTAQ 不是简单 sum_ij (Q_ij-W_ij)^2；它重构输出，因此输入统计已使不同权重的影响不同，并保留行内交叉项。
权重级对角目标 sum_ij (dL/dW_ij)^2 (Q_ij-W_ij)^2 忽略权重误差交叉项；这与 GuidedQuant 的输出加权目标不同。
例如单样本单输出 z=w1*x1+w2*x2，输出梯度为 g：
输出加权误差为 g^2*(dw1*x1+dw2*x2)^2，包含 2*g^2*dw1*dw2*x1*x2。
分别按权重梯度平方加权只剩 g^2*x1^2*dw1^2+g^2*x2^2*dw2^2。
先选清楚目标；本项目默认探索保留行内相互补偿能力的输出误差加权路线。
统一用样本按列：W0、Q 为 [out,in]，Xf、Xq 为 [in,N]，Y0=W0 Xf 为 [out,N]。
W0 是进入当前模块量化前保存的 FP 权重，迭代中的 W 不可覆盖它。
Xf 来自 FP 路径，Xq 来自前序模块已量化的路径；A16 不保证 Xf=Xq。

## 固定通道权重为什么可能无效

若 J(Q)=sum_i c_i ||Q_i Xq-Y0_i||^2，且每行量化可行域独立、c_i>0，
则 argmin_{Q_i} c_i f_i(Q_i)=argmin_{Q_i} f_i(Q_i)。
因此固定通道系数本身不改变精确最优解。共享 scale/位宽预算/码本、非齐次阻尼和有限步优化等可能破坏这一结论的条件。
若代码产生变化，先排查这些因素，而不是立即认定敏感度有效。

## 候选目标：token 与通道两维的权重

J(Q)=1/2 sum_{i,t} s_it (Q_i Xq[:,t]-Y0_it)^2。
取 G_it 为 FP 教师任务损失对该线性输出的梯度，候选 s_it=G_it^2。
这是敏感度/Fisher 启发的替代度量，不是声称恢复精确任务 Hessian。
初始版本固定 G，不在逐列量化时重新计算；与 quant 路径重算梯度是另一个实验。
标准 backward 得到的是指定序列损失对 token 激活的梯度，不应无条件称为逐 token loss 的独立 Fisher 样本。
记录 causal shift、mask、loss reduction、序列长度。平方后平均和平均后平方不等价。

## 单行的 H 和线性项

用列向量 q 表示某一行的权重转置，y=Y0_i.T，D=diag(s_i)。
J_i(q)=1/2 (Xq.T q-y).T D (Xq.T q-y)
       =1/2 q.T H_i q - q.T b_i + const。
H_i=Xq D Xq.T；b_i=Xq D y；gradient=H_i q-b_i。
实现时对 Xq 的列广播 s，避免显式构造 NxN 的 D。
若从当前权重 w 更新 delta，r=y-Xq.T w，则线性项为 Xq D r。
所以加权必须同时进入 Hessian 与残差项；只改 H 不自洽。
若 s 全 1，退化到普通非对称输出重构；再令 Xf=Xq 则回到对称目标。

## 带一个量化约束的参考更新（推导作业）

在当前可更新坐标集合中，定义 J(delta)=1/2 delta.T H delta-b.T delta。
约束 e_k.T delta=a（把第 k 个权重固定到某个量化值所需的差值）。
写拉格朗日函数；由 H delta-b+lambda e_k=0 和约束联立求 delta。
要求：a=0 也可能有残差驱动的其他坐标更新；b=0 要退化到 GPTQ 形式。
使用线性求解而非显式逆矩阵。真实 H 可能奇异；toy 先取满秩输入，后单独测试阻尼。
这仅是当前子问题的参考解；高效 ResComp 的残差分解、已固定列记账和 lazy update 仍须另行验证。

## 分组统计的候选实现合同

对输出组 I_g，s_gt=mean_{i in I_g} G_it^2，保留 token 维。
H_g=Xq D_g Xq.T。
C_g=(Xf-Xq) D_g Xq.T。
T_g=Xf D_g Xq.T=H_g+C_g。
这些是加权输入统计，不能直接断言替换后所有官方快速更新都正确。
先与无 block 的约束参考求解器比较，再迁移到 ResComp 快速实现。
阻尼只加在求解器的 H 上；保留 raw H，不能把阻尼污染进 T=raw H+C。
固定通道权重归一化、下限 epsilon、截断极端梯度和混合强度均须作为显式超参数。

## 边界与研究问题

GuidedQuant 已有输出敏感度加权与分组方法，因此“加权 MSE”本身不是新贡献。
ResComp 修正固定 FP 目标的残差记账；MARR 调节模块的残差强度。
待检验：在公平预算下，任务敏感度加权能否使 ResComp 的补偿更有利于 held-out 任务损失？
尚不能宣称无人做过相同组合，或组合一定优于现有方法。
