"""Write a complete six-round report while retaining the five-round original."""
import json
from pathlib import Path
import statistics as st

root=Path(__file__).resolve().parent
s=json.loads((root/'round6_summary.json').read_text())
passed=[f for f,g in s['continuation_gate'].items() if g['passed']]
transfer=all(c['methods']['block']['vs_gptq']['wins']>=2 and c['methods']['block']['vs_gptq']['ci95'][1]<0 for c in s['layers'].values())
new_judgment=('、'.join(passed)+'通过两层共同的额外收益门槛，值得进一步独立确认。' if passed else
              '本轮token加权、原方向罚项及限制曲率的方向版本均未通过两层共同的额外收益门槛。当前没有证据支持把新增敏感度复杂度设为默认。')
transfer_judgment=('未加权完整块重建在新外部语料、两个目标层上均有支持改善的证据，上一轮的有效方向得到了进一步迁移检验。' if transfer else
                   '未加权完整块的外部迁移表现不满足两个目标层都稳定改善的检查；应保留分层结果，不能写成普遍有效。')
lines=['# 第六轮最终报告：任务敏感度改进、完整块迁移及六轮累计结论','',
       '完成日期：2026-09-27。本报告先给出本轮全部新实验和判断，后附前五轮完整报告。所有结果来自本机实际运行；没有用外部测试成绩重新选参。','',
       '## 1. 本轮结论','',transfer_judgment,'',new_judgment,'',
       'PPL是困惑度，衡量下一词预测，越低越好。这里所有胜负都在相同数据、前缀和目标层内比较；不能拿第五轮WikiText的绝对PPL与本轮PTB数值相减来衡量进步。用白话说，完整块目标让主分支和残差旁路相加后的整体输出接近浮点教师；新敏感度方法则尝试把这个整体误差中更影响任务的部分看得更重。','',
       '原始研究问题仍是：任务敏感度能否改善量化补偿。本轮把敏感度施加到已验证更合理的完整块误差，而不是回到早期单分支目标。同时加入更多优化步数、另一个目标层、外部语料、896参数均值修正及连续权重对照，区分目标选择、计算预算与参数自由度。','',
       '## 2. 已执行的全部改进','',
       '| 方案 | 改动 | 回答的问题 |','|---|---|---|',
       '| block192 / block576 | 未加权完整块，192与576步 | 原方案能否迁移；三倍步数是否值得 |',
       '| token192 / token576 | 以完整块教师梯度平方产生弱token权重 | 原任务加权想法在正确目标上能否恢复价值 |',
       '| direction192 / direction576 | 完整块MSE加一个梯度方向投影误差 | 保留通道间梯度符号关系是否优于仅看幅度 |',
       '| bounded192 / bounded576 | 方向项缩放为9/d，误差空间曲率比限制为10 | 原方向方案是否因过度放大单方向而失效 |',
       '| continuous192 / continuous576 | 同初始化/范围/优化器，去掉round | 固定离散网格是否限制修复，不作为4-bit结果 |',
       '| gptq | 初始量化up参考 | 不做额外修复的起点 |',
       '| bias | 保持GPTQ权重，仅加校准块误差均值 | 896个额外浮点参数是否已经足够 |','',
       '原计划每上下文10个候选，2层×3seed，共60个；开发阶段另追加12个bounded检查点，总72个。五个优化族分别按三个seed平均开发CE选192或576，再加gptq、bias及固定block192/block576锚点。外部共有48个不重复候选条件、6个重复别名；另有浮点和6个前缀参照。与前五轮累计共864次扫描候选执行、258次冻结后的候选评估，执行数包含跨轮重叠，不是独立假设数。','',
       '## 3. 方法、数学和公平比较','',
       'Qwen2.5-0.5B固定revision 060db6499f32faf8b98477b0a26969ef7d8b9987，CUDA/F32、TF32关闭。权重量化范围[-8,7]，每行scale固定为原权重absmax/7。校准每seed64×256，seed=0/1/2；开发64×256。Adam代码空间学习率固定.03、64-token微批次×4累积，所有方法共用完全相同的随机token批次。','',
       '随机token有放回抽样；192/576步分别使用49152/147456次token，约为原校准token量的3/9倍，是重复使用同一校准集而非新增训练数据。因此更长优化可能继续降低校准误差，也可能损害泛化。','',
       '目标为零起算层12或18的mlp.up_proj。两者均使用已保存的前12块GPTQ前缀；层18之前的12–17块保持浮点。它验证同一来源的前缀误差能否在不同位置修复，不是“前18块GPTQ”实验。目标层gate/down和后续块不变，各目标独立运行，不把两个层的收益相加。','',
       '令Bf=原始浮点块输出，Rq=当前前缀下MLP前残差，e_t=Mq(Q)_t−(Bf−Rq)_t。未加权目标L0=mean(e²)。完整块梯度G由原始浮点模型下一词CE对Bf求得；只用校准文本，未使用PTB标签训练。','',
       'token方案：u_t=mean_i(G_it²)，归一化、clip=10、再归一化得到v_t，s_t=.9+.1v_t；损失mean(s·e²)。direction方案：d_t=G_t/max(||G_t||,1e−12)，损失L0+mean_t[(d_tᵀe_t)²]。方向罚项系数固定1；零梯度方向为零。所有目标除以mean((Bf−Rq)²)，保持相同归一化基准。','',
       '方向项展开后含跨通道乘积，保留通道相对符号，但平方后不保留整体梯度方向的一阶增减效应。它不是完整Hessian或严格任务损失，不含跨token项；不同目标仍可能有不同的优化难度。原direction系数固定1，追加bounded系数固定9/d，未进行连续系数网格搜索。若失败，结论限制于已试构造，不能否定所有方向敏感度方法。','',
       '在第一组开发结果中原direction较差，数学核查发现每token误差空间Hessian为2I/d+2uuᵀ，非零单位方向下特征值比d+1=897。因此在外部测试前追加bounded：方向项乘9/d，将这一比值固定为10。它不等于权重空间或真实任务Hessian的条件数。追加版本保持所有训练预算一致，协议和源码修订均保留；属于基于开发结果的自适应迭代，不能声称最初就预注册。','',
       'continuous从相同GPTQ代码初始化，去掉round并裁剪到相同连续范围；它是受限连续权重修复，不保证达到浮点最优。bias为校准误差均值，增加了一个块输出偏置，因此与纯权重方法架构不同、预算也不同。二者只作归因与成本对照。','',
       '## 4. 新外部语料与测试隔离','',
       f"数据来自[PTB公开源文件]({s['source']['url']})，仓库commit `{s['source']['commit']}`，字节SHA256 `{s['source']['sha256']}`。它是预处理的PTB文本，包含原有大小写/未知词等处理；没有将本轮PPL与词级PTB论文成绩比较。",'',
       '按原文本换行、Qwen分词、不添加特殊EOS，划分非重叠256token窗口。四个相邻窗口组成一个1024token块，固定seed=20261001从完整块中选32块，共128窗口、每候选32640计分token。数据准备阶段不计算模型损失，选参仅用原WikiText开发集。','',
       '已检查与历史校准、开发及已用测试的精确窗口哈希无交集；这不保证没有语义重复，也不保证预训练模型未见过公开语料。所有层/方法选择、源码与权重哈希在PTB评估前一起冻结。前一轮73篇WikiText文章没有被重新当作新测试。','',
       '区间计算先平均同窗口三个seed的CE差，再平均每组四窗口，对32个连续块配对bootstrap10000次。块不是独立文章，跨块仍可能相关；因此区间是探索性、条件于当前校准seed的证据，未做多重比较校正。不能把128×3或32640token当作独立样本量。','',
       '## 5. 外部测试完整结果','',
       f"浮点模型在本轮PTB窗口上的PPL={s['floating_point_ppl']:.6f}。以下为三个seed PPL的算术平均；相对变化另由配对平均ΔCE计算。两个目标层是独立条件。",'',
       '| 层 | 方法 | 开发选定配置/固定锚点 | 开发PPL | 外部PPL | 相对GPTQ变化 | 相对所选block变化 |',
       '|---|---|---|---:|---:|---:|---:|']
for layer,c in s['layers'].items():
    for family,m in c['methods'].items():
        lines.append(f"| {layer} | {family} | {m['selected']} | {m['development_ppl']:.6f} | {m['mean_ppl']:.6f} | {m['vs_gptq']['relative_ppl_percent']:+.4f}% | {m['vs_block']['relative_ppl_percent']:+.4f}% |")
lines+=['','新敏感度目标与开发所选未加权block直接比较：','',
        '| 层 | 方法 | ΔCE | 95%块区间 | 赢的seed | 改善的窗口块 |','|---|---|---:|---|---:|---:|']
for layer,c in s['layers'].items():
    for family in ('token','direction','bounded'):
        d=c['methods'][family]['vs_block'];lo,hi=d['ci95']
        lines.append(f"| {layer} | {family} | {d['delta_ce']:+.7f} | [{lo:+.7f}, {hi:+.7f}] | {d['wins']}/3 | {d['improved_blocks']}/32 |")
lines+=['','![第六轮外部效果](figures/round6_effects.png)','',
        '敏感度与未加权方案也需匹配所选步数，避免把预算差异算成任务信息的收益：','',
        '| 层 | 方法 | 相同步数 | 相对同预算block的PPL变化 | 95%块区间（ΔCE） |','|---|---|---:|---:|---|']
for layer,c in s['layers'].items():
    for family in ('token','direction','bounded'):
        item=c['methods'][family];step=item['selected'].split('_s')[-1];d=item['vs_block'+step]
        lo,hi=d['ci95'];lines.append(f"| {layer} | {family} | {step} | {d['relative_ppl_percent']:+.4f}% | [{lo:+.7f}, {hi:+.7f}] |")
lines+=['',
        '## 6. 三倍优化步数、均值修正及连续权重对照','',
        '| 层 | 比较 | ΔCE | 相对PPL | 95%块区间 | 赢的seed |','|---|---|---:|---:|---|---:|']
for layer,c in s['layers'].items():
    for family,reference,label in [('block576','vs_block192','576步 − 192步'),('bias','vs_gptq','均值修正 − 初始GPTQ'),('continuous','vs_block','连续权重 − 所选W4 block'),('block','vs_gptq','所选W4 block − 初始GPTQ')]:
        d=c['methods'][family][reference];lo,hi=d['ci95']
        lines.append(f"| {layer} | {label} | {d['delta_ce']:+.7f} | {d['relative_ppl_percent']:+.4f}% | [{lo:+.7f}, {hi:+.7f}] | {d['wins']}/3 |")
lines+=['','延长步数只改优化预算，不能把它的收益归于敏感度。连续权重如果更好，只说明该训练路径下存在离散网格代价；如果未更好，也不证明离散解超过浮点最优。均值修正若能取得部分收益，说明低频/均值漂移值得关注，但不等于解释完整块方案的全部收益。','',
        '## 7. 是否继续投入及具体优化方向','',new_judgment,'',transfer_judgment,'',
        '| 新方案 | 层 | 至少赢2/3seed | 改善至少0.1% | 区间上界<0 |','|---|---|---|---|---|']
for family,g in s['continuation_gate'].items():
    for layer,checks in g['checks'].items():lines.append('| '+family+' | '+layer+' | '+' | '.join('是' if v else '否' for v in checks.values())+' |')
lines+=['',
        '本轮需要特别注意三个结论。第一，第12层token相对开发选出的576步block改善约0.229%，但固定192步block在同一外部集上的PPL为44.0570，优于token576的44.2653；因此不能把这个局部胜出解释为全面超过旧方案。第二，bounded在第12层的平均差和连续窗口块区间虽支持改善，相对block却只赢1/3校准seed，收益主要集中在一个seed。第三，第18层未加权block和敏感度方案均未稳定超过初始GPTQ，简单bias却在三个seed都改善。','',
        '延长到576步在两个目标层的PTB上都比192步更差，而开发规则仍选中了576步。这是单一开发语料选参的迁移局限，并非可以据外部结果事后改写选择。第六轮据此提供了第七轮“限制权重变化、分离均值漂移、使用两个已见开发域选择”的动机；第六轮测试一旦参与此改进，就不再是第七轮独立证据。','',
        '0.1%是控制研究投入的实用筛选线，不是统计定律；优先结合幅度、种子一致性、语料迁移和计算代价判断。未通过门槛的方案不继续在这份PTB测试上追加参数搜索。','',
        '下一步优先级：先保留本轮跨条件更可靠的简单目标，再考虑两个目标模块的顺序联合修复，检查收益是否抵消；随后扩展完整模型W4和另一模型规模。任何任务加权改进都应超过相同预算的未加权block，而不是只超过原始GPTQ。方向项若保留，应先在新开发语料检查它与真实CE变化是否一致，再考虑系数收缩；本轮没有证明应直接增加更复杂的曲率。','',
        '当前仍缺少打乱token权重/随机方向对照，不能把某个小增益唯一归因为任务信息；新层未重新扫描全部ResComp/GPTAQ族，因此也不能声称超过所有量化方法。本轮没有完成gate/up/down联合量化、完整模型部署或论文查新。若目标是论文主线，应把现阶段定位为误差目标与泛化的实证研究，而不是已经建立全模型新算法优势。','',
        '## 8. 资源安全、成本和验证','',
        '继续执行4 GiB分配器预算、8 GiB整卡停止阈值、78℃停止、4 GiB系统RAM余量，启动需至少5 GiB空闲显存。缓存CPU驻留、单GPU进程顺序执行、64×4微批次、每步/序列40ms间歇；没有提高限额。','',
        '| 指标 | 本轮实测 |','|---|---:|',
        f"| PyTorch峰值分配 | {s['resources']['max_allocated_gib']:.3f} GiB |",
        f"| PyTorch峰值保留 | {s['resources']['max_reserved_gib']:.3f} GiB |",
        f"| 整卡采样最大已用显存 | {s['resources']['max_sampled_total_mib']:.0f} MiB |",
        f"| 采样最高温度 | {s['resources']['max_sampled_temperature_c']:.0f}℃ |",
        f"| 采样最低可用主存 | {s['resources']['minimum_sampled_available_ram_gib']:.2f} GiB |",
        f"| 六组主扫描及六组追加内部计时 | {s['scan_seconds']/60:.2f} 分钟 |",
        f"| 外部评估内部计时 | {s['evaluation_seconds']/60:.2f} 分钟 |",'',
        'GPU整卡约每2秒采样，不是连续峰值；分配器预算独立存在。计时不是完整会话墙钟时间，未覆盖代码编写、下载及全部程序启动。各方法复用缓存，核心时间不含教师梯度采集；单独运行未加权block无需教师梯度，因此敏感度方法的真实额外成本不能只看优化核心秒数。保护降低风险，不能保证排除驱动或硬件故障。','',
        '第12层三个seed的block192检查点与第五轮对应权重逐元素一致，开发CE复现误差小于1e−6；外部评估六次前缀重载也通过同样CE检查。新增方向梯度解析式、等权退化、均值修正最小化及bounded特征值比检查通过。60个W4/偏置候选权重严格符合保存的固定网格，12个连续权重对照明确单列；冻结哈希和开发选参已独立复算。','',
        '本轮公开文本首次下载受到沙箱网络限制，改用授权的只读下载后成功，源文件版本/哈希已固定；这不是实验结果失败。本轮科学结果是否完整，以各目录_summary.complete和上述核查为准。历史黑屏与数值问题详见后附五轮原报告。','',
        '## 9. 每个开发候选及成本','',
        '下面24行覆盖全部72个层/seed候选；耗时为三个seed平均核心优化累计秒数，包含小张量传输、不包含完整模型评估和40ms间歇。bias的一遍拟合计时包含流式统计及间歇，不能与优化核心作严格吞吐比较。','',
        '| 层 | 候选 | seed0 PPL | seed1 PPL | seed2 PPL | 平均校准块MSE | 平均开发块MSE | 计算秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
for layer in ('12','18'):
    runs=[json.loads((root/f'round6_l{layer}_s{seed}/metrics.json').read_text()) for seed in range(3)]
    for seed,r in enumerate(runs):
        bm=json.loads((root/f'round6_bounded_l{layer}_s{seed}/metrics.json').read_text())
        r.update({k:v for k,v in bm.items() if k.startswith('bounded_')})
    for key in runs[0]:
        if key in ('common_prefix','_summary'):continue
        values=[r[key] for r in runs]
        lines.append('| '+layer+' | '+key+' | '+' | '.join(f"{v['ppl']:.6f}" for v in values)+
                     f" | {st.mean(v['calibration_block_mse'] for v in values):.7f} | {st.mean(v['validation_block_mse'] for v in values):.7f} | {st.mean(v['optimization_seconds'] for v in values):.3f} |")
lines+=['','## 10. 产物与复算','',
        '- [本轮事先协议](round6_protocol.md)。',
        '- [机器汇总](../results/round6_summary.json)、[外部结果CSV](../results/round6_summary.csv)、[全部72个开发条件CSV](../results/round6_all_development.csv)。',
        '- [冻结选择与哈希](../results/round6_holdout/selection.json)、[逐窗口外部损失](../results/round6_holdout/metrics.json)。',
        '- [外部数据身份](../data/round6_holdout/manifest.json)、[源码快照](../results/round6_source/source_manifest.json)。',
        '- [前五轮独立保留报告](round5_final_report.md)、[资源说明](../RUN_SAFELY.md)。','',
        '```powershell','python results/summarize_round6.py','python results/write_round6_report.py','```','',
        '以上只重算已有指标，不启动GPU实验。运行入口run_round6.ps1保护已有结果，拒绝覆盖部分扫描或既有外部评估；复跑需要独立输出目录。旧轮脚本并不自动继承本轮保护。','']
text='\n'.join(lines)
(root.parent/'reports/round6_report.md').write_text(text,encoding='utf-8')
history=(root.parent/'reports/round5_final_report.md').read_text(encoding='utf-8')
combined=text+'\n---\n\n# 历史附录：前五轮原始完整报告\n\n以下保留前五轮当时的结果与判断，当前判断以本报告第六轮部分为准。\n\n'+history.split('\n',1)[1]
(root.parent/'reports/round6_final_report.md').write_text(combined,encoding='utf-8')
print('Wrote round6_report.md and round6_final_report.md; passed:',passed,'block transfer:',transfer)
