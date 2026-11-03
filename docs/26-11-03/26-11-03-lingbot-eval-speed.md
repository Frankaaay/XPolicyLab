# LingBot Clean / Random 单卡提速

用户批准独立 branch/worktree、commit/push 到 Frank fork（不合并），仅 H800 集群一张闲时 GPU，持续至重大提升及验证通过。保留原训练工作区和所有既有正式作业。

分支 frank/lingbot-eval-speed，基于现役服务端1a96f3d5165c5663b9c32bfcd19cd116376f3ae2；客户端固定 shared-robotwin-20261023/XPolicyLab 815a2d3，FastWAM evaluator a8a62f8。原 mono checkpoint_step_50000，原25视频/50动作采样步、CFG5/1、2帧chunk、每帧16动作、原冻结seed/instruction/上限均不变。

先运行 tests/h800_lingbot_speed.py run <独立绝对输出目录> --baseline-only。覆盖Clean click_bell/0、Random click_bell/0和自然动态灯光click_bell/19。记录全部内部图像、执行动作、环境RNG、RPC输入输出、编码latent、渲染/仿真/推理/KV/调试写出/empty_cache耗时。代码原有逐块调试写出含同步GPU→CPU拷贝，且每块清理CUDA缓存，待实测判断其占比。

诊断先验证旧路径短回合重复一致。LingBot原服务端使用连续torch RNG且reset不重置随机数，故诊断保存每个基线回合开始时的实际CPU/CUDA/Python/NumPy RNG状态，在该回合重复/候选对照时恢复原状态；不为生产引入新seed策略。同一模型实例固定参数，成对比较所有输入、编码latent、输出、逐步动作/RNG/结果，严格零差异。计时包含相同的诊断哈希开销，首次启动/预热与重复回合分开报告。

当前仅完成静态语法检查和实验准备，没有GPU速度结论。配额research、priority4、ForceQuotaOverSold、1GPU/12CPU/128Gi、最长180min；实际卡型启动后核验（该H800集群以L20Y/Hopper80GiB对外报告）。

## baseline01 故障与基线修正

单卡作业 dlc4wg2kreee6gcs 已 Failed，输出 /mnt/workdir/frank/runs/lingbot-eval-speed-20261103/baseline01。首次推理后 KV 更新报 Float/BFloat16 不匹配，未完成回合。定位到旧1a96f3d遗漏历史修复5a7c8bc3；当前共享服务端815a2d3相对旧版仅增加该dtype修复及输出目录名哈希修复。复用815a2d3服务端作为可运行基线，候选同步相同两项修复；修复不计提速收益。模型、输入、采样、灯光均未改变。下一次 baseline02 使用修正后两次Clean对照及两项Random基线。
