"""Generate a complete seven-round report from verified confirmation metrics."""
import json,datetime
from pathlib import Path

root=Path(__file__).resolve().parent;s=json.loads((root/'round7_summary.json').read_text())
g=s['gates'];hybrid=g['hybrid'];bias=g['bias_only']
if hybrid['extra_over_bias_passed'] and hybrid['all_four_improve_with_2_seeds']:
    decision='均值补偿与受限量化权重修复的组合通过本轮四条件筛选，值得进入有限的多模块/全模型确认；尚不能宣称全模型或新算法优势已成立。'
elif hybrid['all_four_improve_with_2_seeds']:
    decision='组合方案在本轮两个层、两份新语料中均有正向结果，但超过廉价均值补偿的额外优势不够统一。值得有限继续，优先保留简单基线，暂不扩大复杂敏感度设计。'
elif bias['all_four_improve_with_2_seeds']:
    decision='最一致的证据来自低成本均值补偿。额外权重修复尚未在四条件全部稳定胜出，当前更值得保留简单修正，而不是继续扩大任务梯度方案。'
else:
    decision='保守改进仍有跨层或跨语料的不一致，不建议把当前方法直接扩展为全模型主线。保留局部收益，后续投入应限定在失效原因和更独立的确认上。'
lines=['# 最终完整实验报告：七轮迭代、保守修复与研究判断','',
       '报告日期：'+datetime.date.today().isoformat()+'。第六轮运行于9月27日，第七轮运行于9月28日；后文附有前六轮完整报告，保留当时结论与本轮如何修正它们。','',
       '## 1. 最终判断','',decision,'',
       '你的原始想法——利用任务敏感度改善量化误差补偿——工程上可以实现，但七轮证据仍要求区分三件事：梯度加权是否提供额外价值，完整块重建是否泛化，以及简单均值修正能解释多少收益。最后一轮改进重点是限制大容量修复、分离均值漂移和改善选参方式，不再把更多梯度信息默认视为进步。','',
       'PPL是困惑度，越低越好。每个比较都在相同模型、前缀、目标层和数据上进行；不同轮、不同语料的绝对PPL不能直接相减当作进步幅度。','',
       '**最关键的边界：** 本轮hybrid相对初始GPTQ在四条件均改善，且每个条件都是3/3seed；但第18层Shakespeare的95%连续块区间仍跨0。简单bias_only四条件也都是3/3改善，区间均支持负向CE差。hybrid在第12层两语料超过bias_only，在第18层却没有额外优势。因此可以支持有限继续，不能宣称复杂方案全面胜出。','',
       '| 报告简称 | 白话含义 |','|---|---|',
       '| gptq | 已量化的初始权重，作为同条件基线 |',
       '| bias_only | 不改量化码，只加896维输出均值修正 |',
       '| block192 | 用完整块误差训练192步的权重 |',
       '| weight_only | 在初始、半幅和完整修复中，用两个开发域选一个，无额外偏置 |',
       '| hybrid | 用两个开发域选择修复强度，并加对应均值修正 |',
       '| wiki_only_hybrid | 同一组候选，只按WikiText选参，检验选择规则的作用 |','',
       '## 2. 为什么继续修改第六轮方案','',
       '第五轮含残差完整块重建在新WikiText文章上明显改善。第六轮进一步加入两个修复位置、更多步数、弱token加权、方向罚项、受限方向曲率、连续权重与低容量偏置，并在PTB上评估。它同时发现了成功和失效：','',
       '- 第12层固定192步block在PTB的平均PPL为44.0570，初始GPTQ为44.6548，说明原方案有一定外部迁移能力。',
       '- 开发选出的576步block变成44.3675；更多步数并没有更好的外部泛化。第18层block576为44.7781，反而高于初始GPTQ的44.6908。',
       '- 第12层token576相对block576约改善0.229%，但仍差于固定192步block；受限方向方案的平均收益主要由一个seed贡献。不能只拿最有利的比较说任务加权已成功。',
       '- 仅增加896维块输出均值偏置，在第六轮PTB两层都改善，且都是3/3seed；这比增加复杂敏感度更值得优先检查。','',
       '因此第七轮不继续延长训练，而是检验：将已学到的权重修复收缩一半，再独立修正均值，是否能减少对校准域的过度适应。同时用两个已见开发域共同选参，避免只按WikiText排名。','',
       '## 3. 第七轮具体方法与工作量','',
       '沿用固定版本Qwen2.5-0.5B、GPTQ前12块、目标层12/18的mlp.up_proj、校准seed0/1/2各64×256、F32运算和固定逐行W4网格。层18之前的12–17块仍浮点；两个目标独立实验，不代表同时量化或修复两个层。','',
       'q0为第六轮初始GPTQ整数码，q1为192步未加权block的整数码。只试tau=0、0.5、1：Q_tau=scale×round((1−tau)q0+tau q1)。所有候选仍位于原[-8,7]固定网格；半整数采用PyTorch的舍入规则，不把浮点插值权重当作4-bit结果。','',
       '每种权重再比较是否加b_tau=mean_cal(Bf−Bq_tau)，在目标块输出加这个896维向量。偏置只用WikiText校准数据拟合；保存为F32，额外参数量896，原始参数字节数3584，即3.5 KiB，不含框架执行开销。它改变了模型的块输出，必须与纯权重方案分开评价；本轮未测真实int4推理速度。','',
       '对固定候选，令e=Bq−Bf，则均值最优偏置为−mean(e)，校准MSE分解为均值项与中心化误差项之和。偏置能消去均值项，但不保证新数据的任务CE下降。本轮是在已有权重上后拟合偏置，没有重新训练一个“中心化误差”权重目标，也没有联合优化权重和偏置。','',
       '| 候选 | 权重修复 | 块输出偏置 | 来源 |','|---|---|---|---|',
       '| tau0 | 初始GPTQ | 无 | 复用第六轮 |',
       '| tau0_bias | 初始GPTQ | 原均值修正 | 复用第六轮 |',
       '| tau0.5 | 半幅修复后重新取整 | 无 | 本轮新计算 |',
       '| tau0.5_bias | 半幅修复后重新取整 | 重新拟合均值 | 本轮新计算 |',
       '| tau1 | 原192步完整块权重 | 无 | 复用第六轮 |',
       '| tau1_bias | 原192步完整块权重 | 重新拟合均值 | 本轮新计算 |','',
       '6种候选×2层×3seed=36条开发条件，其中18条新计算、18条明确复用旧权重与成绩。旧初始GPTQ还在两开发域逐seed复算CE检查。没有把36条全部计成新训练或独立假设。所有新权重只是对已有192步结果后处理，本轮没有额外Adam训练。','',
       '## 4. 如何选配置，如何保留真正未见的确认数据','',
       '第六轮PTB test成绩已经看过，本轮将那128个窗口明确转为开发数据；原WikiText validation64窗口是另一开发域。对每层、每候选，在每域先平均三个seed的CE差（候选减初始GPTQ），再取两域差值中较差的一个，选择使它最小的共同配置。','',
       'weight_only从三个无偏置强度选择；hybrid从三个偏置候选和初始GPTQ回退项选择。回退项在两域的差值都为0，避免强行选择开发阶段已有明显负作用的方法。两个族都不按每个seed分别挑最好参数。此规则只保障已观察开发均值，不保证新域或每个seed必然改善。','',
       '| 层 | 方案 | 冻结的共同配置 | Wiki开发ΔCE | 已见PTB开发ΔCE | 较差域ΔCE |',
       '|---|---|---|---:|---:|---:|']
for layer,c in s['development'].items():
    for family in ('weight_only','hybrid','wiki_only_hybrid'):
        key=c['selected'][family];d=c['gaps'][key]
        lines.append(f"| {layer} | {family} | {key} | {d['wiki']:+.7f} | {d['ptb_seen']:+.7f} | {c['worst_domain_gap'][key]:+.7f} |")
lines+=['','wiki_only_hybrid是确认前追加的选择器消融：使用与hybrid完全相同的候选池，只看Wiki开发CE。没有新增权重、训练或超参数；若与hybrid选择相同，外部结果即为别名，不能说双域选择贡献了改进。','',
        '新确认数据全部在上述选择和权重哈希冻结后才计算模型指标：','']
for name,source in s['source'].items():
    lines.append(f"- **{name}**：[固定版本来源]({source['url']})，commit `{source['commit']}`，SHA256 `{source['sha256']}`。")
lines+=['','PTB使用此前未评估的官方validation文件，在本项目中充当新确认集，不能叫成标准官方test成绩。Shakespeare是本轮首次评估的新文本来源。两者均按Qwen分词，选32个连续1024-token块，每块分4个非重叠256窗口，各128窗口、32640计分token。','',
        '已经检查与旧校准、开发及已用测试的精确token窗口哈希无交集，也检查两个新集互不重叠。不能由此保证没有语义重复，或预训练模型没见过这些公开文本。32块不是32篇独立文章，长距离文本依赖仍可能存在。','',
        '统计先在同窗口平均3seed差值，再平均每4窗口块，配对重采样32块10000次。区间为探索性，未做多重比较校正；不能把token数、窗口数乘seed数当独立样本量。','',
        '## 5. 两份新确认集完整结果','',
        '原始浮点模型参考PPL：'+'；'.join(f"{dataset}={c['floating_point_ppl']:.6f}" for dataset,c in s['datasets'].items())+'。下表是前缀量化加单目标修复，并非完整模型W4。','',
        '| 语料 | 层 | 方法 | 冻结配置 | 平均PPL | 相对GPTQ变化 | 赢的seed |','|---|---|---|---|---:|---:|---:|']
for dataset,c in s['datasets'].items():
    for layer,methods in c['layers'].items():
        for family,m in methods.items():
            d=m['vs_gptq'];lines.append(f"| {dataset} | {layer} | {family} | {m['selected']} | {m['mean_ppl']:.6f} | {d['relative_ppl_percent']:+.4f}% | {d['wins']}/3 |")
lines+=['','PPL为各seed PPL的算术平均；相对PPL变化为exp(平均配对ΔCE)−1，与两个算术均值相除略有差别。gptq对自身的胜出数为0，不表示失败。若所选配置与某个固定参考相同，则复用该评估并明确是别名，不算额外独立模型。','',
        f"确认阶段共{ s['evaluation']['unique_candidate_contexts'] }个不同的候选/层/seed组合，在两份新文本上各评估一次；加两个浮点参考，共{ s['evaluation']['actual_dataset_evaluations'] }次完整数据集评估。报告另有同一权重的方案别名，不能据此增加独立实验数量。",'',
        '![第七轮新确认结果](figures/round7_effects.png)','',
        '## 6. 组合修复是否真的超过廉价均值补偿','',
        '| 语料 | 层 | hybrid − bias_only ΔCE | 相对PPL变化 | 95%连续块区间 | 赢的seed |',
        '|---|---|---:|---:|---|---:|']
for dataset,c in s['datasets'].items():
    for layer,methods in c['layers'].items():
        d=methods['hybrid']['vs_bias_only'];lo,hi=d['ci95']
        lines.append(f"| {dataset} | {layer} | {d['delta_ce']:+.7f} | {d['relative_ppl_percent']:+.4f}% | [{lo:+.7f}, {hi:+.7f}] | {d['wins']}/3 |")
lines+=['','筛选要求比bias_only至少改善0.1%、区间上界<0且至少赢2/3seed，并且在两层×两新语料四条件中都成立。这是控制投入的实用门槛，不是统计定律。没有达到时，应优先考虑偏置的低成本，不能只报告组合相对初始GPTQ的收益。','',
        '| 方法 | 四条件均改善GPTQ且至少赢2/3seed | 四条件区间均支持改善GPTQ | 四条件都满足相对bias的额外收益门槛 |',
        '|---|---|---|---|']
for family,d in g.items():lines.append('| '+family+' | '+' | '.join('是' if d[k] else '否' for k in ['all_four_improve_with_2_seeds','all_four_ci_support','extra_over_bias_passed'])+' |')
lines+=['',decision,'',
        '同候选池的双域选择与只看Wiki选择的直接对照：','',
        '| 语料 | 层 | 双域hybrid − Wiki-only ΔCE | 相对PPL | 95%连续块区间 |','|---|---|---:|---:|---|']
for dataset,c in s['datasets'].items():
    for layer,methods in c['layers'].items():
        d=methods['hybrid']['vs_wiki_only_hybrid'];lo,hi=d['ci95']
        lines.append(f"| {dataset} | {layer} | {d['delta_ce']:+.7f} | {d['relative_ppl_percent']:+.4f}% | [{lo:+.7f}, {hi:+.7f}] |")
lines+=['',
        '## 7. 逐seed结果及全部开发条件','',
        '| 语料 | 层 | 方法 | seed0 PPL | seed1 PPL | seed2 PPL |','|---|---|---|---:|---:|---:|']
for dataset,c in s['datasets'].items():
    for layer,methods in c['layers'].items():
        for family in ('gptq','bias_only','weight_only','hybrid','wiki_only_hybrid','block192'):
            lines.append('| '+dataset+' | '+layer+' | '+family+' | '+' | '.join(f'{x:.6f}' for x in methods[family]['seed_ppl'])+' |')
lines+=['','以下每行平均三个seed，12行覆盖36条开发条件；逐seed原表见CSV。','',
        '| 层 | 候选 | Wiki开发PPL | 已见PTB开发PPL | 较差域ΔCE | 来源 |','|---|---|---:|---:|---:|---|']
for layer,c in s['development'].items():
    for key,p in c['mean_ppl'].items():
        provenance='复用第六轮' if key in ('tau0','tau0_bias','tau1') else '本轮新增'
        lines.append(f"| {layer} | {key} | {p['wiki']:.6f} | {p['ptb_seen']:.6f} | {c['worst_domain_gap'][key]:+.7f} | {provenance} |")
lines+=['','量化码与偏置的实际改动（先对每个seed计算，再取三seed均值）：','',
        '| 层 | 候选 | 相对初始GPTQ改变的码占比 | 平均绝对码移动 | 偏置RMS |',
        '|---|---|---:|---:|---:|']
for layer,candidates in s['parameter_changes'].items():
    for key in ('tau0_bias','tau0.5_bias','tau1_bias'):
        d=candidates[key]
        lines.append(f"| {layer} | {key} | {d['changed_code_percent']:.3f}% | {d['mean_absolute_code_move']:.5f} | {d['bias_rms']:.6f} |")
lines+=['','tau=0.5先在整数码之间插值再取整，因此不保证恰好一半的权重改变，也不保证任务效果恰好减半。偏置RMS只描述参数大小，不能单独解释CE收益。','',
        '## 8. 最后的推断与后续投入','',
        '**第12层：保留完整块修复有依据。** hybrid在PTB新文件和Shakespeare上相对GPTQ分别改善约1.624%和0.841%，相对bias_only仍分别改善0.860%和0.213%，满足本轮该层的额外收益条件。这里双域与Wiki-only选中了相同配置，不能给双域选择记一份不存在的独立收益。','',
        '**第18层：收缩减少了过度修复，但偏置更划算。** 原block192在两个新集平均都比GPTQ差，Shakespeare甚至3/3seed都变差。半幅加偏置变成四项中的本层两项3/3seed正向，但相对bias_only，PTB约差0.0085%且区间跨0，Shakespeare约差0.1072%且区间支持退化。额外改动约16.52%的量化码，仍没有超过896维均值修正，不能认为这份权重优化成本已经合理。','',
        '**选择器消融提供了具体证据。** 同候选池下，第18层双域选出的半幅方案，比只看Wiki选出的完整幅度方案，在PTB新文件和Shakespeare分别改善约0.305%和0.277%；两者均3/3seed、配对区间全负。这支持本次保守跨域选择，但只有一个产生不同选择的目标层，仍不能声称该规则普遍最优。','',
        '**下一轮的有限投入建议。** 固定当前候选与选参规则，不再扩大网格。首先验证两个模块顺序组合，并以每层都只加偏置作为廉价对照；同时检查一个不同模型或目标位置，使用新的确认数据。“第12层用hybrid、第18层只用bias”是根据本轮结果提出的后续假设，尚未联合运行或独立确认，不能把它当成已经交付的最优完整模型。若这些验证仍不能稳定超过bias_only，应收缩复杂权重修复路线。','',
        '已经得到的证据：任务加权H/C可以实现；局部MSE下降不保证任务改善；把残差纳入块目标可以产生明显局部收益，但更多优化、更多敏感度结构和换层并不会自动泛化。第六轮外部负结果推动了第七轮收缩修复、均值分离及双域选参，这些改变都在新确认成绩之前固定。','',
        '第七轮改变了后处理和选参两个因素，因此不能把总体差异唯一归因于权重收缩；它也没有直接证明任务梯度权重有效。均值偏置的闭式拟合与大容量权重修改有不同开销和模型形式，必须分别报告。新确认集现在已被查看，后续不能继续把它们称为未见测试。','',
        '若继续：先以初始GPTQ、bias_only和本轮冻结的保守方案作为固定基线，验证两个模块顺序组合是否相互抵消；再做完整模型W4与更大模型。仍需新的确认数据、足够的校准seed，以及真实int4/F16部署检查。若增加任务敏感度，应给未加权方案相同预算并加入打乱权重/随机方向对照，不再以弱默认基线证明优势。','',
        '不建议马上做的事：继续在已看过的PTB/Shakespeare上加密tau、rho、方向系数和步数；扩大GPU预算；将局部固定网格结果写成整模型加速；在未做文献核查时声称算法新颖性。当前所有模型仍以F32保存假量化权重，没有交付packed int4推理模型。','',
        '## 9. 温度保护、资源调整和实际成本','',
        '第七轮初始40ms间歇运行在一次监测到79℃时被78℃保护停止；当时整卡显存3940 MiB、系统可用RAM约13.22 GiB，没有记录OOM。未完成目录及源码单独保留，不混入正式结果。','',
        '随后增加每序列间歇到150ms。启动降温条件先试60℃，在61–62℃多次等待；调整启动条件时主动结束了一次已经进入模型加载、但没有完整候选的进程，该记录也单独保留。最终启动温度要求≤65℃，温度硬停止仍为78℃，显存分配预算仍4 GiB、整卡停止8 GiB、RAM余量至少4 GiB；没有提高这些硬限制，也没有修改显卡驱动或系统功耗。','',
        '| 指标 | 本轮成功运行实测 |','|---|---:|',
        f"| PyTorch峰值分配 | {s['resources']['max_allocated_gib']:.3f} GiB |",
        f"| PyTorch峰值保留 | {s['resources']['max_reserved_gib']:.3f} GiB |",
        f"| 整卡采样最大显存 | {s['resources']['max_sampled_total_mib']:.0f} MiB |",
        f"| 采样最高温度 | {s['resources']['max_sampled_temperature_c']:.0f}℃ |",
        f"| 采样最低可用主存 | {s['resources']['minimum_sampled_free_ram_gib']:.2f} GiB |",
        f"| 成功开发流程内部计时 | {s['scan_seconds']/60:.2f} 分钟 |",
        f"| 两新集确认运行时间估计（含补跑） | {s['evaluation']['evaluation_seconds']/60:.2f} 分钟 |",'',
        '最终确认接近结束时，写resources.json触发OSError Errno 22并退出；当时已保存71条指标，只缺3条。中断现场单独存档，resume_round7.py重新核对全部冻结哈希和当前前缀，仅计算缺失项，随后逐项确认71条原指标完全不变。没有重选配置或重训练。保存改用临时文件、文件替换及有限重试；日志无法确定此次文件写入错误的底层原因。','',
        '首次确认的精确计时因异常退出未保存，表中按初始资源采样到最后指标写入估计该阶段时长，再加精确记录的补跑时长；含初始化、不含两次运行之间的空闲间隔。合并资源记录缺少失败前约31秒尚未落盘的采样，报告峰值仅代表已保存记录。','',
        '计时包含本轮对应阶段的低负载间歇，不是纯算法吞吐，也不包括完整会话、所有下载/编写、启动等待和中断成本。整卡温度/显存是采样最大值，不是连续峰值；这些保护降低风险，不能保证排除所有硬件或驱动问题。','',
        '核查：36个候选严格位于保存的固定网格，并逐一复算收缩公式；18个复用条件与父检查点的权重/偏置逐位一致；冻结源码/配置/权重哈希一致；双域选择独立复算通过；71条中断前指标不变；新数据身份与开发窗口隔离；每次前缀重载和初始GPTQ复算采用1e−6 CE阈值。新增收缩端点/码值约束及双域选择回退的数学检查通过。','',
        '完整回归共24项测试通过，涵盖已有量化数学、分组/退化端点、迭代目标、方向曲率、收缩及选择器规则。测试不证明泛化能力，方法质量以新语料结果为准。运行记录见[测试日志](../results/round7_tests.log)。','',
        '## 10. 报告与证据入口','',
        '- [第七轮事先协议及资源修订](round7_protocol.md)。',
        '- [第七轮汇总JSON](../results/round7_summary.json)、[确认结果CSV](../results/round7_summary.csv)、[完整开发CSV](../results/round7_all_development.csv)。',
        '- [全部冻结选择](../results/round7_holdout/selection.json)、[逐窗口新确认指标](../results/round7_holdout/metrics.json)。',
        '- [新确认集身份](../data/round7_holdout/manifest.json)、[源码快照](../results/round7_source/source_manifest.json)。',
        '- [仅缺失项补跑记录](../results/round7_holdout/resume_manifest.json)、[原中断日志](../results/round7_interrupted_holdout_write/interrupted.log)。',
        '- [白话技术说明：原理、实现、结果阅读与文件导航](../docs/TECHNICAL_WALKTHROUGH.md)。',
        '- [第六轮报告](round6_report.md)、[前五轮原报告](round5_final_report.md)。',
        '- [安全运行说明](../RUN_SAFELY.md)、[79℃中断日志](../results/round7_stopped_temperature_l12_s0.log)。','',
        '在Experiment目录复算已有统计和报告，不会启动GPU模型：','',
        '```powershell','python results/summarize_round7.py','python results/write_round7_report.py','```','',
        '运行入口run_round7.ps1对未完成目录和既有确认结果拒绝覆盖；复跑请使用独立输出位置。第六轮/第五轮报告生成器现在分别写历史报告，不覆盖本最终报告。','']
text='\n'.join(lines)
(root.parent/'reports/round7_report.md').write_text(text,encoding='utf-8')
history=(root.parent/'reports/round6_final_report.md').read_text(encoding='utf-8')
combined=text+'\n---\n\n# 历史附录：前六轮完整报告\n\n以下保留前六轮证据及当时判断；当前结论以第七轮前文为准。\n\n'+history.split('\n',1)[1]
(root.parent/'reports/final_report.md').write_text(combined,encoding='utf-8')
print('Wrote round7_report.md and complete seven-round final_report.md.')
print(decision)
