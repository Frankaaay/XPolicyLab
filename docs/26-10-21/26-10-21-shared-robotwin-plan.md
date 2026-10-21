# LingBot 接入共享 RoboTwin 冻结 plan

基线d3b75e0，保留原文档修改，未commit/push或部署。

新增policy/LingBot_VA/lingbot_va/evaluation/robotwin/shared_plan_policy.py供FastWAM-C2R共同环境入口加载。manifest显式model=lingbot及policy={host,port,stereo,video_guidance_scale,action_guidance_scale}；真正的server推理参数以原job_config为准，正式联调仍须核验。旧eval_polict_client_openpi.py直接运行会退出并指向共同入口。

保留WebSocket推理、首次动作帧跳过、四keyframe/帧、原action反馈KV cache。训练数据xyzw相对EEF在环境边界转成绝对wxyz；拒绝无效动作及空chunk。原生图像为RGB，不引入颜色转换；StereoTok显式读取三路独立右目。达到共同step limit或成功即停止，禁止超限执行。

服务器CPU作业dlcwiv8qzjqbkhpn的5项测试通过：四元数/锚点、原chunk执行节奏及KV反馈、步数上限、独立右目、无效动作。尚未真实GPU/SAPIEN联调，不能声称已与另外两个模型生成相同真实场景。完整计划与记录见C:/Project/StereoTok/docs/26-10-21/26-10-21-robotwin-shared-plan.md。

## 授权发布与单卡验收

2026-10-21 用户明确允许提交/推送本次变更和1张闲时GPU串行验收。具名提交保留其他dirty文件。Fast测试入口tests/check_shared_plan_runtime.py仅用于真实Clean/Randomized初始化和两次原生策略调用，不生成正式分数；固定索引LF换行，LingBot空动作步维度拒绝。实际远端结果另行追加。
