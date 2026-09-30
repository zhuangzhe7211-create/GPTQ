# 实验文本

`wikitext2_20260926_s0/s1/s2` 来自同一 WikiText-2 raw 数据，只改变文档排序的随机 seed。

- `calibration.jsonl`：来自 train，采集输入和梯度。
- `validation.jsonl`：来自 validation，比较候选方案。正式三次实验都使用 s0 的验证文件。
- `manifest.json`：来源、数据指纹、文档数和文件 SHA256。

程序从文档内切固定长度窗口，数量由运行参数决定，不是把整个 JSONL 都用于每次实验。文件名中的 s1/s2 不代表新的独立数据集。

第二轮 test 直接从本地缓存的官方 test split 读取；实际使用的行号和窗口哈希另存于 `results/round2_test/test_manifest.json`。没有将 test 混进这里的校准文件。

第三轮的 `round3_holdout/` 是从未用于校准的128篇官方train文章构成的预留评估集，**不是官方test split**。`tokens.pt` 为每篇一个256-token窗口，`manifest.json` 记录文章、原始行、窗口哈希及排除的136篇校准文章。数据提前划分，但只在验证选参全部冻结后计算模型损失。
