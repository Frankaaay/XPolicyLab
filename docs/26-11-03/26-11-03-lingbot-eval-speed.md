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
