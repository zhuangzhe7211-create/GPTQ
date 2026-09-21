# 本次验证记录

日期：2026-09-21。

* Python 3.14.0、PyTorch 2.13.0+cpu、Transformers 5.15.0。
* test_quant_core：6/6 通过。涵盖加权统计与残差、Cholesky 融合、全一分组回归、输入重标度等价、单大块参考内核一致、常数缩放及零梯度回退。
* 随机小 Llama：教师梯度 → 公共 RTN 前缀 → 五个候选 → 验证 CE/PPL 完整运行通过。
* 随机小 Qwen2：同样完整运行通过；最终代码运行记录位于 results/smoke_qwen2_verified。
* 所有新增 Python 文件语法检查通过。

尚未验证：CUDA 执行、预训练 Qwen 权重下载/评估、WikiText 数据下载、正式全模型量化、独立测试集收益。

随机模型结果没有研究质量意义，不能作为超过 ResComp 的证据。核心基线取固定官方源码，但整套模型/数据管线是本地单模块 pilot。
