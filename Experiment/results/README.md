# 原始实验产物

读结论请先去 [第三轮累计报告](../reports/round3_report.md)，也可回看 [第一轮](../reports/20260926_report.md) 和 [第二轮](../reports/20260926_round2_report.md)；这里用于复核数字。

| 目录或文件 | 含义 |
|---|---|
| `smoke_qwen2_verified/` | 整理前保留的历史随机模型验证 |
| `20260926_cuda_smoke/` | 本机第一次 CUDA 流程验证 |
| `20260926_qwen05b_w4_s0/s1/s2/` | 第一轮真实模型实验 |
| `20260926_summary.json` | 第一轮三次结果汇总 |
| `summarize_20260926.py` | 重新生成第一轮汇总 |
| `round2_smoke/` | 第二轮新增流程的随机模型检查 |
| `round2_s0/s1/s2/` | 第二轮公平调参和打乱对照 |
| `round2_test/` | 验证集选参冻结后的独立测试 |
| `round2_summary.json` / `summarize_round2.py` | 第二轮汇总与重算脚本 |
| `round3_l6_s0/` 等9个目录 | 第三轮：三个目标模块×三个校准seed的扫描 |
| `round3_holdout/` | 全部选择冻结后对新预留文章的评估 |
| `round3_logs/` | 第三轮逐次运行的控制台日志 |
| `round3_summary.json` / `.csv` | 第三轮结果的机器汇总和表格 |
| `summarize_round3.py` / `plot_round3.py` | 重算汇总和报告图形 |
| `round3_source/` | 本轮源码快照及哈希；包含零权重端点修复前后版本 |
| `round3_failed_l12_s2/` | 全一权重FP32舍入回归失败的原始记录；不是完成的实验 |
| `round3_null_diagnosis.json` / `round3_endpoint_regression.json` | 数值问题定位及修复兼容性检查 |
| `round4_prefix_s0/s1/s2/` | 三个校准 GPTQ 前缀及哈希，不是完整模型 |
| `round4_rtn_attn_s0/` 等12个目录 | 第四轮：两种前缀×两种模块×三个seed，216个扫描候选 |
| `round4_holdout/` | 冻结选择、独立新文章评估及原始逐窗口指标 |
| `round4_summary.json` / `.csv` | 第四轮主结果、区间与预设继续门槛核对 |
| `summarize_round4.py` / `write_round4_report.py` | 从原始指标重算统计、图及中文报告 |
| `round4_source/` | 第四轮核心源码、诊断修复与评估修复的版本快照 |
| `round4_failed_rtn_mlp_s0_diagnostics/` | 大梯度分位数诊断失败记录，不是完成实验 |
| `round4_failed_holdout_prefix_device/` | CPU/CUDA前缀复现检查失败及最初冻结选择 |
| `round4_prefix_device_diagnosis.json` | CPU/CUDA格点差异与原验证CE复现证据 |
| `round4_verification.json` | 216个候选完整性与42个历史权重逐位复现检查 |
| `round5_rtn_s0/` 等6个目录 | 第五轮初始目标与较小补偿系数扫描，每组35个候选 |
| `round5_block_rtn_s0/` 等6个目录 | 根据开发结果追加的含残差完整块目标，每组6个候选 |
| `round5_holdout/` | 完成所有开发实验后统一冻结选择，再评估73篇新文章 |
| `round5_resource_preflight/` | 低显存短检查，只是资源检查，不是质量证据 |
| `round5_interrupted_rtn_s0/` | 黑屏报告前的未完成扫描及当时源码，不混入正式统计 |
| 每个第五轮目录中的 `resources.json` | 分配器峰值、整卡显存/温度采样与主存余量 |
| `round5_source/` / `round5_environment.json` | 原始及修订版本快照、哈希与环境版本 |
| `round5_summary.json` / `.csv` | 第五轮全部选定配置、区间、资源峰值与继续门槛 |
| `round5_verification.json` / `verify_round5.py` | CPU独立检查246份权重网格、冻结哈希、选参复算及数据身份 |
| `summarize_round5.py` / `write_round5_report.py` | 重算统计、图与包含五轮详细回顾的最终报告 |
| `round6_l12_s0/` 等6个目录 | 第六轮主扫描，两个层×三个seed，每组10候选 |
| `round6_bounded_l12_s0/` 等6个目录 | 受限方向曲率追加，每组2候选；在外部评估前完成 |
| `round6_holdout/` | 统一冻结选择后的PTB test逐窗口结果；第七轮将其转为已见开发集 |
| `round6_summary.json` / `.csv` / `round6_all_development.csv` | 第六轮72候选审计、外部成绩、资源与预设门槛 |
| `round6_source/` | 原协议、方向曲率追加版本与分析代码的源码哈希 |
| `round7_l12_s0/` 等6个目录 | 第七轮36条开发条件；18条新计算、18条明确复用第六轮 |
| `round7_holdout/` | 两域选参及Wiki单域选择对照全部冻结后，两份新文本的确认结果 |
| `round7_summary.json` / `.csv` / `round7_all_development.csv` | 第七轮配对统计、全条件表、网格/来源审计与资源峰值 |
| `round7_stopped_temperature_l12_s0/` 及同名 `.log` | 79℃触发保护的未完成尝试；不混入正式结果 |
| `round7_interrupted_start_l12_s0/` 及同名 `.log` | 修改启动降温条件时主动中断的尝试；不算质量实验 |
| `round7_interrupted_holdout_write/` | 确认阶段写资源日志失败时的71条原指标、冻结选择及日志 |
| `round7_holdout/resume_manifest.json` / `round7_resume.log` | 仅补跑3条缺失项，71条已有指标逐项不变 |
| `round7_tests.log` / `round7_verification.json` | 24项实现测试及最终产物、链接、快照核查 |
| `round7_source/` | 运行、资源保护、确认前选择器消融及最终分析的版本快照 |
| `summarize_round6.py` / `write_round6_report.py` | 重算第六轮汇总并写历史报告，不覆盖最新总报告 |
| `summarize_round7.py` / `write_round7_report.py` | CPU复算第七轮统计、图和七轮完整总报告 |

`round2_test/selection.json` 保存看到 test 前选出的设置和权重哈希。`test_manifest.json` 保存实际测试文本身份。`metrics.json` 保存逐窗口结果和汇总。

其他实验目录的 `config.json`、`diagnostics.json`、`metrics.json` 和 `.pt` 的含义见 [主入口](../README.md)。原始数字不手工修改；每次重跑用新目录。
