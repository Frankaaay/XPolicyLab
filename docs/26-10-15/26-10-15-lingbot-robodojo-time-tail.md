# LingBot RoboDojo 时间与尾部合同

2026-10-15。main@34d6135d609d46a10ccb81da24d5ddad0b4c1f7a，LingBot 子树与交接 f6e309e 无差异；未提交。用户批准 RoboDojo25Hz→stride2/真实12.5Hz，两路线均8动作/latent，不截尾。配置名称 robotwin30_train 是本适配器已有名称，这次不改 RoboTwin 原始帧率。

process_data 保留全部 stride2 原采样帧，补齐至足够容纳全部动作的1+4k规则索引，实际像素重复末帧，记录 frame_valid；视频、parquet、metadata长度不一致直接报错。target_fps 严格检查实际帧率，默认12.5，deploy action_per_frame=8。dataset 将prefix/tail动作设mask=false，保留每个原动作；纯padding latent透传mask到既有视频loss。旧无padding元数据的latent使用全有效mask。

CPU unittest 2项通过：遍历1..513长度，原采样帧无丢失，N×30动作有效计数完整，8动作/latent，padding动作零且mask=false。Python compile、diff检查通过；未运行训练/backward/optimizer。真实全3500条转换、模型前向及缺失右目仍待验收，测试不构成全量训练准备完成。

StereoTokAutoencoder 复用 FastWAM clean b856e4929c6a552540008808b188729361fa531a 的完整冻结 student，严格 u8000 SHA。WanVAEStreamingWrapper 保存每个 wrapper 独立的 encoder 和 geometry 缓存，首块1帧、后续4帧，reset 同时清空两组缓存；离线 process_data 与 wan_va_server 使用同一接口。双目必须显式提供三相机映射，拒绝左目代替右目，部署在原 T-shape 布局前分别编码各相机。生产数据集缺 latent 改为直接报错，不静默筛掉 episode。

远端现有 lingbot-va-baselines-20261014 环境补 timm==1.0.20（--no-deps，未改变其他包），同步 install.sh 与 requirements.txt。真实 RoboTwin wrist9帧生产 wrapper 检查：输出 posterior [1,96,3,8,10]，与整段 FP32 编码最大误差0、reset最大误差0，全部冻结，17.54秒。报告 /mnt/workdir/frank/runs/eighteen-preparation-20261014/lingbot-production-streaming-forward.json。此为真实无更新前向，尚未完成生产 joint transformer、全量 latent 转换与最终训练配置验收。

原LingBot环境已用 --no-deps --no-build-isolation 注册上述既有 clean FastWAM dependency；无需临时sys.path。process_data 要求新空输出目录和 v2.1，拒绝覆盖源或已有产物，缺视频直接失败。全部修改尚未commit/push，正式全量处理等待发布SHA；未启动训练。

发布授权：2026-10-15 用户明确确认创建 Frankaaay/XPolicyLab fork，在 frank/eighteen-preparation 分支仅提交任务清单15个文件。基线34d6135d609d46a10ccb81da24d5ddad0b4c1f7a。本提交记录此前验证和本次生产接入，不宣称全量latent已生成或训练已就绪；后续正式运行绑定本提交SHA，继续完成准备。
# 原生全量并行生产

用户批准LIBERO采用新左+新右，Wan对照使用同一新左，不再要求匹配发布旧图。新增utils/render_libero_stereo.py：逐个恢复记录states[i]，forward后同状态重取proprio并渲染两对相机/metric depth/K/外参，不step动作、不逆积分、不依赖states[i+1]；actions[i]原样保留，明确pre-action派生合同。包含全部原始帧和末帧，以.partial输出并在完整task结束后重命名。相机局部X基线6cm/2cm，渲染前后FULLPHYSICS逐值不变。此生产适配仍需真实完整episode验证后再批量运行。

新增process_native_hdf5.py仅用于原生RoboTwin及LIBERO缺失缓存，复用process_data的VAE/text编码及tail mask。RoboTwin使用已准入2500条manifest的左目、xyzw EEF16及官方stride4/T形布局；LIBERO使用raw HDF5全部轨迹、垂直翻转至官方图像方向、action7/stride1/128分辨率。fps标签沿用作者50/60，与物理仿真频率不作等同声明。各并行区间重新编号为独立LeRobot数值+latent数据集，由原MultiLatentLeRobotDataset组合；原HDF5保留且source-episodes.json记录逐条映射，不创建重复视频或声称包含MP4。LIBERO Long直接复用既有官方缓存。

用户确认：StereoTok正式路线改用u15853，原生路线仍用各模型原生Wan VAE；不得用StereoTok覆盖原生缓存。u15853 SHA256为61c6057b49bec9c52f96793a20247d3746978939f0551c37142bdbda6b252eae。

用户要求原生九路线优先，复用既有缓存，缺缓存使用闲时GPU并行。process_data.py新增episode-start参数，与既有num-episodes组合选择不重叠连续区间；每个进程使用独立空输出目录，保留原episode索引、采样和编码逻辑。分片不能直接当完整训练集，完成后合并元数据及各episode文件链接，再使用统一训练目录。计划3500条分4份，每份875条，不改变数据语义。


### 新双目实际执行结果

`4c9f4a52e49f13ec1805e7aab78e88a63f5cdcce` 在个人 clean remote clone 实际跑完 LIBERO object/demo_0，132 帧用时 8.584 秒，actions 保持不变，FULLPHYSICS 前后状态和双目基线检查通过。初次 EGL vendor 环境遗漏已通过复用现有 system-libs 配置解决，无依赖变更。四 worker 全量 40 task/2000 episode 渲染已在 DSW 启动，PID 231336；输出 `/mnt/workdir/frank/runs/eighteen-preparation-20261014/libero-fresh-paired-20261015/full`。一次健康检查各 worker 均持续完成 episode，GPU 60%，全量尚未完成。新输出为 opencv 图像方向，下游读取不可继续无条件垂直翻转。运行没有 backward/optimizer。

十二项准备：LingBot codec 绑定选定 u15853；process_native_hdf5 增加对应 checkpoint 参数、真实右目读取、新 LIBERO opencv 图像方向及缓存 tokenizer_binding。AST 通过；依赖 FastWAM 新 loader 发布，未启动本轮全量编码或 GPU smoke。
