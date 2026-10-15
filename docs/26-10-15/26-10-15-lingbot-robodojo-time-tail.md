# LingBot RoboDojo 时间与尾部合同

2026-10-15。main@34d6135d609d46a10ccb81da24d5ddad0b4c1f7a，LingBot 子树与交接 f6e309e 无差异；未提交。用户批准 RoboDojo25Hz→stride2/真实12.5Hz，两路线均8动作/latent，不截尾。配置名称 robotwin30_train 是本适配器已有名称，这次不改 RoboTwin 原始帧率。

process_data 保留全部 stride2 原采样帧，补齐至足够容纳全部动作的1+4k规则索引，实际像素重复末帧，记录 frame_valid；视频、parquet、metadata长度不一致直接报错。target_fps 严格检查实际帧率，默认12.5，deploy action_per_frame=8。dataset 将prefix/tail动作设mask=false，保留每个原动作；纯padding latent透传mask到既有视频loss。旧无padding元数据的latent使用全有效mask。

CPU unittest 2项通过：遍历1..513长度，原采样帧无丢失，N×30动作有效计数完整，8动作/latent，padding动作零且mask=false。Python compile、diff检查通过；未运行训练/backward/optimizer。真实全3500条转换、模型前向及缺失右目仍待验收，测试不构成全量训练准备完成。

StereoTokAutoencoder 复用 FastWAM clean b856e4929c6a552540008808b188729361fa531a 的完整冻结 student，严格 u8000 SHA。WanVAEStreamingWrapper 保存每个 wrapper 独立的 encoder 和 geometry 缓存，首块1帧、后续4帧，reset 同时清空两组缓存；离线 process_data 与 wan_va_server 使用同一接口。双目必须显式提供三相机映射，拒绝左目代替右目，部署在原 T-shape 布局前分别编码各相机。生产数据集缺 latent 改为直接报错，不静默筛掉 episode。

远端现有 lingbot-va-baselines-20261014 环境补 timm==1.0.20（--no-deps，未改变其他包），同步 install.sh 与 requirements.txt。真实 RoboTwin wrist9帧生产 wrapper 检查：输出 posterior [1,96,3,8,10]，与整段 FP32 编码最大误差0、reset最大误差0，全部冻结，17.54秒。报告 /mnt/workdir/frank/runs/eighteen-preparation-20261014/lingbot-production-streaming-forward.json。此为真实无更新前向，尚未完成生产 joint transformer、全量 latent 转换与最终训练配置验收。

原LingBot环境已用 --no-deps --no-build-isolation 注册上述既有 clean FastWAM dependency；无需临时sys.path。process_data 要求新空输出目录和 v2.1，拒绝覆盖源或已有产物，缺视频直接失败。全部修改尚未commit/push，正式全量处理等待发布SHA；未启动训练。

发布授权：2026-10-15 用户明确确认创建 Frankaaay/XPolicyLab fork，在 frank/eighteen-preparation 分支仅提交任务清单15个文件。基线34d6135d609d46a10ccb81da24d5ddad0b4c1f7a。本提交记录此前验证和本次生产接入，不宣称全量latent已生成或训练已就绪；后续正式运行绑定本提交SHA，继续完成准备。
