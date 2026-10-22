# 共享RoboTwin真实联调：LingBot缓存dtype修复

用户要求持续完成当前已授权验收，1张闲时GPU串行，不扩大到全量测评。

v6 dlcem1rm391wybl6：FastWAM Wan和OpenWAM StereoTok均完成Clean/Randomized两次策略调用，初始七路RGB/场景/seed/instruction完全等于初始化基准，有限16D EEF动作已实际执行。LingBot StereoTok已成功生成第一动作块并执行，错误发生于首块后的_compute_kv_cache反馈（并非模型未能加载或首个动作生成失败）。FP32编码/归一化结果直接进入BF16 patch_embedding_mlp，报mat1 Float / mat2 BFloat16；action随后也跟随latent的dtype。

定向修复：仅在KV反馈进入DiT之前，将latent_model_input转为self.device/self.dtype，action沿既有to(latent)路径同步。保留VAE/归一化FP32、原生BF16 DiT和推理步数/guidance/数据/权重，不增加autocast或修改全模型精度。与_infer原有BF16噪声及conditioning边界一致。增加first-frame与后续chunk测试，核验cast数值、action dtype、原编码dtype未改以及帧计数。AST解析通过；实际Torch测试和真实LingBot联调待运行。

## v7回归结果与checkpoint可用性

v7 dlc1adiuq3axs753的6项LingBot测试全部通过，新增KV边界测试覆盖初始帧与后续chunk。随后复用旧checkpoint时complete.json不存在，按检查停止，未退回不完整权重。v8 dlcicp9mz7mmz7e5改为启动时重新挑选当前完整checkpoint并记录实际step/complete marker，保留其余推理设置和修复SHA；仅用于代表性接口验收。没有宣称v7复现使用了同一权重，没有将新旧权重结果混成分数。先前已通过Fast/Open结果保留。v8请求和脚本前缀shared-runtime-lingbot-v8。

## v8连续反馈定位与修复

v8 dlcicp9mz7mmz7e5使用完整step31000，KV dtype修复已通过首轮反馈；后续原生反馈为8帧，StereoTok wrapper只接受4帧而退出。仅在流式编码器内按完整4帧组顺序编码并拼接latent，保留全部帧、因果缓存和原生动作节奏；首组仍为1帧。增加整9帧及1+8分组与既有1+4+4逐元素完全一致测试，非法尾组拒绝。AST通过，实际Torch回归及连续真实推理由v9执行。

## 最终验收通过（2026-10-22）

v9 `dlckkotu3pv3ga54` 已实时查询为 **Succeeded**。日志同时出现 `SHARED_RUNTIME_ACCEPTANCE_COMPLETE`、`NATIVE_MODEL_ACCEPTED lingbot`、`THREE_NATIVE_MODELS_SHARED_RUNTIME_ACCEPTED_NO_FORMAL_SCORE`。9项真实LingBot环境回归测试全部通过，其中mono真实VAE小模型的1+8和整9帧编码与既有1+4+4结果零容差一致。StereoTok真实checkpoint_step_31000在Clean/Randomized各完成2次策略调用、48个有限16D EEF动作及连续KV反馈，没有dtype/分组异常；测试进程正常退出，服务清理后作业Succeeded。

与v6的FastWAM Wan、OpenWAM StereoTok产物汇总逐项核对：两种phase的seed、instruction、scene、全部7路RGB和step_limit均完全一致。远端 `/mnt/workdir/frank/runs/shared-robotwin-acceptance-20261021/accepted-models.json` 已生成，分项为fastwam-model.json/openwam-model.json/lingbot-model.json。完整本机证据为上述请求目录的shared-runtime-lingbot-v9-full-logs.json、lingbot-model-v9-result.json。

验收代码：FastWAM a8a62f82598e0cc3c47426451fd4fda690c3198a，OpenWAM 11abfa5bf382c04457ae9824c48fb472ec502527，XPolicyLab b8164454bd75b5b7d703e72f28a6c89336e0b52c，均已推送到本任务分支。固定plan SHA256为42ba67ecad191a5910a0c783569509eeaede422819db5cb6e792915e6743f84e；入口核验100份冻结plan，真实联调选择adjust_bottle两种phase的ordinal0。

范围：三种代表模型的共享仿真/计划及连续原生推理接口验收通过，不代表全10,000 episodes成绩、不代表全部六个VAE组合均完成GPU验收。模型动作不同导致后续轨迹/观测不同是正常闭环结果；对齐的是相同初始场景、指令、随机状态和环境执行规则。原生推理步数/guidance/观测与动作节奏保持不变。本轮未更改既有训练或全量测评作业。测试结果中的inference_seconds列表存在跨phase引用，故本次不作为速度统计证据。
