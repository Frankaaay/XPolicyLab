# LingBot RoboTwin 完整恢复修复

## 目的和边界

修复 Wan 的完整状态恢复失败，同时覆盖 StereoTok 共用入口。不改变模型、数据、训练预算、优化器参数或 W&B lineage，不允许退回仅加载权重。保留用户已确认的两份最新完整 checkpoint 策略。

本地基线：`frank/eighteen-preparation`，`3db7c891dade5da7079dfd9d71acd47ef32bdbf7`；原远端训练使用 `1b8e56d71c91613aa42984f59a161bbddc52f15e`。已有 `docs/26-10-16/26-10-16-lingbot-full-resume.md` 的无关未提交修改保留。

## 真实失败与根因

Wan 作业 `dlci75sejcdynb7c` 恢复时报 `KeyError: condition_embedder_action.text_embedder.linear_1.weight`。完整状态路径：`/mnt/workdir/frank/runs/eighteen-preparation-20261014/twelve-ready-u15853/lingbot-official-pair-20261019/lingbot-robotwin-wan/checkpoints/checkpoint_step_11000/training_state.pt`。

只读诊断 `dlch3efpnepzrf2a` 使用部署环境 Torch 2.9.0+cu126 的 FakeTensorMode/mmap，确认839个 optimizer group 参数对应835个已初始化 AdamW 状态；缺失的是 action text_embedder 两层的 weight/bias。模型 forward 使用共享 `condition_embedder.text_embedder`，所以这4个参数没有梯度，AdamW 合法地不创建状态。保存 step=11000，scheduler last_epoch=11000，8个 rank 的数据位置均 epoch=35、batches=45。

Torch 2.9 DCP `_split_optim_state_dict` 对每个 requires_grad 参数直接访问其 state，触发异常；不是 checkpoint 已丢失动量。StereoTok 共用同一实现，也受影响。

## 定向修复

恢复时先校验 optimizer 参数组数量、参数名和顺序。仅在调用 DCP loader 期间，将保存状态中没有 AdamW 条目的参数临时标记为不需梯度，避免加载器创建虚假动量或索引不存在的条目；finally 恢复 requires_grad。已有参数的动量、step 和组参数仍从完整状态加载。后续 scheduler、全局 step、各 rank RNG、epoch 和 DataLoader cursor 的恢复保持原实现。

扩展现有 `tests/test_lingbot_full_resume.py`：模型同时含始终未使用和恢复后才开始使用的参数；检查缺失状态保持缺失、requires_grad 恢复，并比较不中断与恢复后训练的批次、loss、FP32权重、AdamW、scheduler、进度。同步原测试中过期的保留1份断言为现行保留2份。

## 验证与部署状态

本地 AST 和 git diff --check 通过。本地没有 torch，不能替代远端数值测试。

8卡小模型测试 `dlc1vbcxzfp68ssy` 等待资源，主动取消，缩为2卡闲时测试 `dlcn76wp7keo3a17`，仍待调度。测试隔离路径：`/mnt/workdir/frank/runs/lingbot-lazy-resume-test-20261020/`；命令为部署环境 Python 执行 `torch.distributed.run --standalone --nproc_per_node=2 .../tests/test_lingbot_full_resume.py`。实验候选代码只用于测试，未替换正式生产源码。

CPU生产恢复代码定向测试 `dlcsqyscr7zrtiqb` 在参数组直接相等比较处失败；进一步诊断 `dlc1eo2pl7exmd5g` 确认唯一差异是 DCP 将 betas tuple 转成 list，值一致。生产恢复逻辑已在实际 Torch 2.9.0+cu126 上验证：恢复前2个 active 状态，恢复后仍2个，后续激活 later 分支后两边均4个状态；3次后续更新的权重及全部 AdamW tensor/step逐值相等，unused状态始终未创建，requires_grad全部恢复。日志两份参数组逐字段核对仅容器类型差异；GPU测试的递归比较已允许等值tuple/list。

该CPU检查不覆盖FSDP分片、CUDA RNG或多进程DataLoader；这些必须等待2卡完整测试。正式 Wan 尚未恢复，StereoTok 正在旧实现上运行；修复还未提交、推送或部署。正式发布后应核验 Wan 从11000完整恢复、原W&B `dce5ab0d`，StereoTok 原W&B `c11cf986`，且后续抢占启动使用相同修复版本。当前仓库要求 commit/push 明确授权；本次修复准备好后需用户确认发布，不能拿此前仅限评测修复的推送授权扩大到训练代码。

## 看板和实时快照

## 发布授权

用户后续明确允许提交推送本次训练恢复修复；要求完整恢复验证通过后才恢复 Wan，并部署 StereoTok 的恢复修复且保留当前进度。后续发布限本次源码、测试和本记录，不提交已有无关 dirty 文档。多卡测试尚未通过前，不启动正式 Wan 续训，也不停止正在运行的 StereoTok。

2026-10-20 09:45只读快照 `dlc4vxpfvum2vjf7`：OpenWAM StereoTok 23370/30000、2.489秒/步；OpenWAM Wan Clean 4065/5000、3604成功（88.66%，阶段分数）、最近1小时185回合、errors为空；LingBot StereoTok 330/50000，尚无完整 checkpoint。Wan 保存10000与11000完整状态，恢复作业失败。

LIBERO 已经通过40任务原生结果与2000唯一回合交叉验收：Wan1960/2000=98.00%，StereoTok1962/2000=98.10%。飞书大表更新至revision163，保留原论文列和既有表结构；阶段结果与最终结果明确区分。
