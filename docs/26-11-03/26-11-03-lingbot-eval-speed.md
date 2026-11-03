# LingBot Clean / Random 单卡提速

用户批准独立 branch/worktree、commit/push 到 Frank fork（不合并），仅 H800 集群一张闲时 GPU，持续至重大提升及验证通过。保留原训练工作区和所有既有正式作业。

分支 frank/lingbot-eval-speed，基于现役服务端1a96f3d5165c5663b9c32bfcd19cd116376f3ae2；客户端固定 shared-robotwin-20261023/XPolicyLab 815a2d3，FastWAM evaluator a8a62f8。原 mono checkpoint_step_50000，原25视频/50动作采样步、CFG5/1、2帧chunk、每帧16动作、原冻结seed/instruction/上限均不变。

先运行 tests/h800_lingbot_speed.py run <独立绝对输出目录> --baseline-only。覆盖Clean click_bell/0、Random click_bell/0和自然动态灯光click_bell/19。记录全部内部图像、执行动作、环境RNG、RPC输入输出、编码latent、渲染/仿真/推理/KV/调试写出/empty_cache耗时。代码原有逐块调试写出含同步GPU→CPU拷贝，且每块清理CUDA缓存，待实测判断其占比。

诊断先验证旧路径短回合重复一致。LingBot原服务端使用连续torch RNG且reset不重置随机数，故诊断保存每个基线回合开始时的实际CPU/CUDA/Python/NumPy RNG状态，在该回合重复/候选对照时恢复原状态；不为生产引入新seed策略。同一模型实例固定参数，成对比较所有输入、编码latent、输出、逐步动作/RNG/结果，严格零差异。计时包含相同的诊断哈希开销，首次启动/预热与重复回合分开报告。

当前仅完成静态语法检查和实验准备，没有GPU速度结论。配额research、priority4、ForceQuotaOverSold、1GPU/12CPU/128Gi、最长180min；实际卡型启动后核验（该H800集群以L20Y/Hopper80GiB对外报告）。

## baseline01 故障与基线修正

单卡作业 dlc4wg2kreee6gcs 已 Failed，输出 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/baseline01。首次推理后 KV 更新报 Float/BFloat16 不匹配，未完成回合。定位到旧1a96f3d遗漏历史修复5a7c8bc3；当前共享服务端815a2d3相对旧版仅增加该dtype修复及输出目录名哈希修复。复用815a2d3服务端作为可运行基线，候选同步相同两项修复；修复不计提速收益。模型、输入、采样、灯光均未改变。下一次 baseline02 使用修正后两次Clean对照及两项Random基线。

## 第一轮候选

修正基线作业 dlc1dunp57e8qs7f，SHA7271b4f1013413cce151c91edddffef3e86ae140，输出 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/baseline02，服务端已就绪。首轮候选仅调整 wan_va_server.py：遵守现有请求save_visualization=False，默认仍写出；移除_infer和_compute_kv_cache末尾empty_cache，保留_reset清理。AST语法及diff检查通过，实际infer方法在reset/KV/infer及默认/显式开启共5项轻量分派检查通过；尚无GPU候选验证/速度结果。不改采样/输入/RNG/模型。

## baseline02 流式反馈修复

第二次KV收到原客户端完整8帧时，旧StereoTok封装仅接受4帧而退出。复用已有b8164454连续多组流式反馈修复，同时用于两臂公共编码器；保留全部输入帧和原连续缓存，不计加速收益。已完成2次推理：首次9.542s，第二次7.135s；调试写出分别9.35ms/3.62ms，empty_cache约1.3ms，非主要直接开销。KV首次0.317s。继续完整回合验证；运行输出仍保留baseline02。

## 第二轮候选：缓存槽位管理

第一轮成对作业 dlc1w5yayz15qg4h，SHAffdd4995846e3652bf96021e6bea6b362999e69e，输出 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/paired01，当前加载中。第二轮将WanAttention的id/mask/is_pred放CPU，K/V仍在原GPU；容量不足时显式在原K/V设备argsort，保留不稳定排序的原设备同值顺序。诊断按baseline/candidate切换原/新init_kv_cache、allocate_slots，每次reset重建缓存，确保基线不包含该优化。既有诊断中增加80次真实方法操作对照，覆盖容量不足/部分淘汰/临时回退/预测清理；CPU DSW环境测试通过，GPU同值顺序验证将在下一单卡作业启动前执行，未通过则不启动测评。静态语法/diff检查通过，真实回合和GPU验证尚待运行。

## 单卡推理分片开销

第一轮Clean严格对照通过，64动作/18观察/6 RPC；冷基线82.788s，热重复70.462s，候选70.542s，当前无实质端到端收益，不把冷/热差当提速。进一步定向修改_configure_model：仅eval_mode且world_size=1跳过FSDP，训练和多卡继续原分片；5项分派测试通过。第二轮诊断同时加载原FSDP和候选普通Transformer（同一GPU、相同checkpoint），逐参数验证推理BF16精度下权重一致，第二次加载前后恢复torch RNG；按变体切换模型。VAE/文本编码器共享，模型额外占用在80GiB单卡内验证。两臂KV方法独立选择，回合输入/latent/动作/RNG逐项验证，不降低精度或采样。

## 第二轮单卡实测启动

第一轮Random静态64步通过精确对照：75.946→74.190s（1.024x），推理19.975→19.950s，仍无重大提升。为避免继续消耗GPU验证已知低收益组合，停止本轮诊断dlc1w5yayz15qg4h，API确认Stopped；动态灯光尚未完成，不声称已通过。保留全部partial输出。更新隔离clone至af51c19a5ac81109092c7a34b3b1f1f329225b0e，提交第二轮dlc19eztajlbf14m，输出 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/paired02，同一闲时配额单卡限制，先GPU缓存80操作与推理权重门禁，后Clean/Random完整对照。

paired02已通过CUDA缓存80操作门禁及839个参数张量的推理精度权重一致门禁，server ready后进入Clean。此时尚无第二轮完整回合速度结论。

## 第二轮Clean重大推理提升

63动作/17观察/6 RPC全部严格一致，含每次观察、编码latent、动作、环境与模型RNG。热基线69.391820s，候选57.558541s，端到端1.206x/耗时下降17.05%；推理20.151484→8.602011s，2.343x。模型RPC合计20.763939→8.913139s。原诊断candidate comparison引用首次冷基线81.630756s并显示1.418x，该数不能作热态收益；本地诊断修正为若存在已通过的repeat则用repeat作候选参照，并写单独warm comparison。后续Random尚待完成。

第二轮全部短回合对照通过。Random静态64步74.692844→63.134861s（1.183x），推理20.377916→8.482921s（2.402x）；动态58步74.304687→61.081541s（1.216x），推理20.379868→8.232435s（2.476x）。全部输入/latent/动作/RNG/成功结果严格一致。为覆盖真实长轨迹缓存淘汰，在既有tests诊断增加--long-cases：保留Clean click_bell/0作热态门禁，随后原冻结Clean open_microwave/99和Random open_microwave/1，原生上限/seed/instruction不变。继续单卡闲时，短回合结果不足以声明完整5000回合耗时。

paired02作业API已Succeeded，GPU释放。长回合补充作业dlc1ty9zbfy279vh，精确SHA7a525a2175d5106bac0a12bd2c59d1dc92e6cdc8，输出 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/long01，既有测试run附加--long-cases，仍1GPU/priority4/ForceQuotaOverSold/180min。提交前确认旧作业终止及远端clone clean后ff更新，没有重叠GPU占用。

## 长回合Clean通过

long01再次复核Clean短63步热基线74.194404→63.377022s，推理21.472956→9.133512s，严格一致。Clean open_microwave/99完整1500步（均失败）883.862998→631.767469s，1.3990x，节省28.522%；推理410.697500→165.892356s，2.4757x；RPC424.796893→172.772679s。376观察、96 RPC、所有动作/环境与模型RNG/输入/latent/结果严格一致，覆盖实际72帧缓存窗口之后的淘汰。渲染202.577466→202.289454s、仿真219.460763→219.964693s基本不变，收益来自推理。Random长回合继续运行，当前不能声称全套完成。

## 闲时资源回收及定向补跑

long01的Random基线完整1500步883.428333s、推理381.852209s、376观察，均失败。候选完成24 RPC后停止推进；API确认dlc1ty9zbfy279vh Failed，ReasonMessage为pod being preempted/evicted，非代码/仿真错误。候选已完成前缀逐项一致，但没有完整结果，禁止用其估计整回合耗时。保留全部产物。既有诊断增加--random-long-only，仅复核短Clean热态门禁及Random open_microwave/1完整成对，不重复已完成Clean长回合；原始模型RNG起点只保存在旧进程内，故不能把旧基线和新进程候选拼接，必须重新成对。新诊断将每个基线起点的实际CPU/CUDA/Python/NumPy RNG保存到输出目录，以保留资源回收后的复现证据；没有改变生产随机数规则或模型代码。继续原单卡闲时配额，不提高优先级或改用固定卡。

补跑SHA9db6c6a7df1841ef8761c33f21dbbd791e8cd24a，作业dlc1ld68ejadxg34，输出 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/random-long02，run --random-long-only。远端GitHub直连停滞/超时，核验/proc cwd和cmdline后仅终止自己卡住的git-remote-https请求；临时SSH反向转发至本机已有HTTP代理后，同一origin正常fetch并ff，随后转发随SSH退出关闭。未改变origin、分支、目标SHA或使用旁路worktree；提交前核验远端clean且HEAD匹配。

## 最终验收与汇总

Random定向补跑dlc1ld68ejadxg34已由API确认Succeeded；本次测速GPU作业结束。其余诊断基线失败/主动停止/闲时回收状态均保留历史，不改既有正式任务。仅使用H800集群一张闲时GPU，实际设备报告NVIDIA L20Y/Hopper 9.0、81559MiB。全程没有并行占用第二张卡、提升优先级或使用固定资源。

| 固定回合 | 动作/结果 | 热态基线秒 | 候选秒 | 端到端加速 | 推理秒（旧→新） | 推理加速 |
|---|---|---:|---:|---:|---|---:|
| Clean click_bell/0，seed100000 |63/成功|69.391820|57.558541|1.2056x|20.151484→8.602011|2.3426x|
| Random click_bell/0，seed100000，静态灯光 |64/成功|74.692844|63.134861|1.1831x|20.377916→8.482921|2.4022x|
| Random click_bell/19，seed100019，动态灯光 |58/成功|74.304687|61.081541|1.2165x|20.379868→8.232435|2.4756x|
| Clean open_microwave/99，seed100132 |1500/失败|883.862998|631.767469|1.3990x|410.697500→165.892356|2.4757x|
| Random open_microwave/1，seed100001，静态灯光 |1500/失败|867.604315|632.836522|1.3710x|384.726765→156.071808|2.4651x|

以上5个代表回合共3185动作、803观察、210 RPC，成对的全部观察/模型输入/编码latent/原始输出动作/执行动作/环境与模型RNG/初始场景及成功结果严格相同。两个长回合均保留原生1500步失败结果，覆盖真实72帧KV窗口后的淘汰。短回合Clean取已通过重复门禁的热态repeat，绝不使用初次冷启动收益。每个回合在同一作业内成对，跨作业数字不拼接。两项长回合分别节省28.522%与27.059%，均约少4分钟。

Random长回合RPC397.221476→162.422100s，渲染228.248654→228.033918s、仿真210.144315→210.034308s，明确表明主要收益来自推理。原25/50采样步、CFG5/1、checkpoint_step_50000、所有KV图像帧、冻结seed/instruction和灯光规则不变。

验证：每次GPU作业均通过80次缓存操作精确对照（含部分淘汰同值排序、临时恢复与预测清理）及839参数张量推理精度核验；CPU同类80操作、5项infer分派和5项configure分派检查通过，后者覆盖单卡eval/单卡train/多卡及未初始化情形。静态语法与diff检查通过。主生产改动局限于单卡eval跳过FSDP、CPU槽位元数据（保留GPU原排序）、遵守已有save_visualization及减少逐块empty_cache；两项旧兼容修复已单列且不计提速。

证据：短回合paired02（测试SHAaf51c19a5ac81109092c7a34b3b1f1f329225b0e）；Clean长回合long01（7a525a2175d5106bac0a12bd2c59d1dc92e6cdc8）；Random长回合random-long02（9db6c6a7df1841ef8761c33f21dbbd791e8cd24a）。目录均在 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/，保留contract.json、cache-check.json、model-weight-check.json、*-candidate[-warm]-comparison.json、逐步JSON/JSONL及日志。后两测试SHA相对af51c19仅调整测试/文档，生产代码一致。

这是固定代表回合的单卡性能验证，不是全部50任务×100回合的正式成绩或全量时长；多卡吞吐未在本轮验证。独立分支frank/lingbot-eval-speed推送Frank fork，不合并、不修改现有正式评测作业。本轮目标已达到，停止继续占用GPU优化。
