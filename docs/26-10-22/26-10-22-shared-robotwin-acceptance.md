# 共享RoboTwin真实联调：LingBot缓存dtype修复

用户要求持续完成当前已授权验收，1张闲时GPU串行，不扩大到全量测评。

v6 dlcem1rm391wybl6：FastWAM Wan和OpenWAM StereoTok均完成Clean/Randomized两次策略调用，初始七路RGB/场景/seed/instruction完全等于初始化基准，有限16D EEF动作已实际执行。LingBot StereoTok已成功生成第一动作块并执行，错误发生于首块后的_compute_kv_cache反馈（并非模型未能加载或首个动作生成失败）。FP32编码/归一化结果直接进入BF16 patch_embedding_mlp，报mat1 Float / mat2 BFloat16；action随后也跟随latent的dtype。

定向修复：仅在KV反馈进入DiT之前，将latent_model_input转为self.device/self.dtype，action沿既有to(latent)路径同步。保留VAE/归一化FP32、原生BF16 DiT和推理步数/guidance/数据/权重，不增加autocast或修改全模型精度。与_infer原有BF16噪声及conditioning边界一致。增加first-frame与后续chunk测试，核验cast数值、action dtype、原编码dtype未改以及帧计数。AST解析通过；实际Torch测试和真实LingBot联调待运行。
