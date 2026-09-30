# 第二轮：公平调参、梯度来源与分组成本

本协议在查看第二轮结果前写定。沿用第一轮 Qwen2.5-0.5B 固定 revision、FP32/CUDA、W4、目标 layers.12.self_attn.o_proj、公共 RTN 前缀、校准 seeds 0/1/2，每次 32×256 tokens，共享 64×256 验证窗口。

## 问题和候选

第一轮加权收益小，未超过 GPTAQ 参考平均值，并且量化前缀造成较大输入漂移。本轮比较以下方法族，每族仅在 alpha={0.25,0.5,1.0} 中选一个值，统一按三个校准 seed 的平均验证 CE 选择：

| 方法族 | 变化 | 组数 | rho |
|---|---|---:|---:|
| rescomp | 普通官方模块核心 | 1 | 0 |
| teacher | 第一轮浮点教师梯度 | 4 | 0.5 |
| prefix | 已量化公共前缀、目标权重仍为原始浮点时采集梯度 | 4 | 0.5 |
| shared | 浮点教师梯度、全输出共享 token 权重 | 1 | 0.5 |
| gptaq | GPTAQ 神经元分解参考，beta=0 | 1 | 0 |

固定 alpha2=0.25（仅 ResComp），不修改 vendor 算法、不做训练、不改变目标 W0@xf。不得根据 test 结果再选参数。本轮调整对象是量化方法和验证流程，保持预训练模型本身不变。

## 对照、选参和独立测试

- 额外记录各校准 seed 下 teacher、prefix、shared 在 alpha=0.25 的 3 次打乱权重对照，打乱 seed=100/101/102。此对照用于观察任务对齐，不参与方法族选择。
- 保留第一轮固定 alpha=0.25 的 teacher 与 rescomp，在独立 test 同样评估，区分调参收益与方法变化。
- 首先写出 `selection.json`：每个方法族在平均验证 CE 上选出的 alpha，同时冻结选中权重的 SHA256。之后才加载 test 文本并评估。
- test 来自 WikiText-2 raw 的官方 test split，seed=20260926 随机排序，排除校准/验证 JSONL 中完全相同的文本和 token 窗口；每个合格文本行只取首个 256-token 窗口，共 128 个窗口。记录原始行号、文本哈希、token 哈希。
- 每个候选在相同 test 窗口上计算 CE/PPL。报告每个校准 seed 和三次均值，保留负结果。不把三个 seed 当成三个独立测试集。
- 显存和耗时仅作本机描述，记录梯度采集与量化内核耗时；分组数=1 是成本优化候选，不预设效果更好。
- 测试后停止本轮调参。结论仅覆盖这个单模块、模型、位宽和数据协议。

## 运行

在 `C:\GPTQ\Experiment` 下设置本地缓存和离线模式，然后运行：

```powershell
$env:HF_HOME='C:\GPTQ\Experiment\.hf_cache'
$env:HF_HUB_OFFLINE='1'
foreach ($pilotSeed in 0,1,2) {
    python -u run_pilot.py --refine --model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --device cuda --dtype float32 --calibration "data/wikitext2_20260926_s$pilotSeed/calibration.jsonl" --validation data/wikitext2_20260926_s0/validation.jsonl --target model.layers.12.self_attn.o_proj --samples 32 --eval-samples 64 --seq-len 256 --bits 4 --groups 4 --rho 0.5 --seed $pilotSeed --out "results/round2_s$pilotSeed"
    if ($LASTEXITCODE -ne 0) { throw 'Pilot failed' }
}
python -u evaluate_refinement.py
```
