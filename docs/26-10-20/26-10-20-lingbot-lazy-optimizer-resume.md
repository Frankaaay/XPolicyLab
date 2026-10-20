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

## 发布授权

用户后续明确允许提交推送本次训练恢复修复；要求完整恢复验证通过后才恢复 Wan，并部署 StereoTok 的恢复修复且保留当前进度。后续发布限本次源码、测试和本记录，不提交已有无关 dirty 文档。多卡测试尚未通过前，不启动正式 Wan 续训，也不停止正在运行的 StereoTok。

## 2026-10-20 发布与GPU验证

修复已发布到个人远端 `frank/eighteen-preparation`：`1a96f3d5165c5663b9c32bfcd19cd116376f3ae2`。只包含train.py、既有测试及本记录；原26-10-16 dirty文档未提交。

2卡验证因资源排队，尝试原地缩为1卡被平台拒绝（租户不支持 edit/inplace restart），未生效。随后停止2卡请求，提交单卡 `dlc3s251nsts2t8m`，Succeeded。实际Torch2.9 FSDP2、fused AdamW、bf16、2个随机worker、GA2、跨epoch，保存2步后继续4步；与不中断训练相比batches/loss/FP32权重/AdamW/scheduler/step/epoch全部精确一致，再次保存与恢复也通过。包括永久unused参数和恢复后激活参数，未伪造lazy state。结果 `/mnt/workdir/frank/runs/lingbot-lazy-resume-test-20261020/output/result.json`。单卡不冒充8卡验证；准备的Wan正式请求内先运行同版本8卡测试，任何失败阻止正式续训。

独立clean clone由 `dlc17geo7f2oirlf` 创建并验证：`/mnt/workdir/frank/projects/lingbot-resume-1a96f3d/XPolicyLab`。原运行clone未修改。新基线包含RoboDojo新增的默认 `stereotok_input_mode=stereo`，旧checkpoint合同没有该字段；仅此固定双组wrapper显式断言并移除该新增默认字段，使缓存latent训练的有效配置保持原样。正式源码仍严格比较全部恢复合同，不放宽校验。安装作业 `dlcx6v3m39df9g7p` 将比较旧/新全部配置以及真实checkpoint合同，成功后才更新shared launcher/source SHA；原wrapper/spec/audit具名备份。StereoTok当前进程不会停止或重置。

安装实测通过：两路各42项合同配置与旧版相等，Wan真实11000 checkpoint合同完全相等，日志 `REAL_CHECKPOINT_CONTRACT_PASS`、`EXACT_PREVIOUS_CONFIG_PASS`、`PAIR_RESUME_FIX_INSTALLED`。只改运行入口的source_root/source_sha及上述已断言的旧配置字段处理；参数、数据、checkpoint和W&B不变。读回审计 `dlc1w5fxu3qezisj` 再次验证wrapper/spec SHA以及两路源版本。

新Wan作业 `dlc1jny7m3kmf96h` 已提交，各8GPU/96CPU/768Gi、priority4、ForceQuotaOverSold、AIMaster Sync/100次restart/ExitCodeAndErrorMsg/nohang。请求在本机既有 `h800-requests/frank-rolling-two-20261018/lingbot-wan-full-resume-fixed-20261020.json`，回执同名前缀。每次启动先在唯一 `resume-verification/wan-UUID` 输出目录执行已发布测试的8卡版本，断言result passed/world8/fsdp2；失败即退出，不执行正式训练。通过后才选择最新完整checkpoint（不得低于11000），调用原wrapper，W&B仍必须online/resumed/dce5ab0d。该门槛已做AST检查，实际8卡测试及Wan完整模型加载尚待资源调度，不能称已恢复运行。

StereoTok作业 `dlcir4z637o2wnrk` 未停止，保留内存中当前训练与原W&B c11cf986（再次从实际日志核验online）。其原UserCommand每次AIMaster启动重新读取shared spec/audit并选择最新完整checkpoint，因而下次恢复使用修复clone；当前已导入的老进程继续训练和保存相同格式。没有人为抢占演练或重置当前进度。

## 看板和实时快照

2026-10-20 09:45只读快照 `dlc4vxpfvum2vjf7`：OpenWAM StereoTok 23370/30000、2.489秒/步；OpenWAM Wan Clean 4065/5000、3604成功（88.66%，阶段分数）、最近1小时185回合、errors为空；LingBot StereoTok 330/50000，尚无完整 checkpoint。Wan 保存10000与11000完整状态，恢复作业失败。

LIBERO 已经通过40任务原生结果与2000唯一回合交叉验收：Wan1960/2000=98.00%，StereoTok1962/2000=98.10%。飞书大表更新至revision163，保留原论文列和既有表结构；阶段结果与最终结果明确区分。

09:58最终审计：Wan新作业Queuing；StereoTok738/50000、最近120步2.442秒/步，仍无首个1000步完整checkpoint。OpenStereo23630/30000、2.514秒/步；OpenWanClean4095/5000、3613成功=88.23%、近1小时195回合、errors为空。LIBERO再次读取2000唯一回合，1960/1962成功未变。飞书大表已按这些证据更新，Wan明确写排队/8卡验证通过后自动续训；StereoTok注明当前训练未中断及恢复入口已修复，不将短窗口ETA当作稳定速度。
