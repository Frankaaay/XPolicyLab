# LIBERO双目评测适配

用户授权补齐OpenWAM/LingBot LIBERO双目评测并提交推送、上云测试，先两个单卡闲时代表抽样，正常后六组单卡全量。

统一复用XPolicyLab.utils.render_libero_stereo.FreshStereoEvalEnv与训练prepare_xml：6cm主相机/2cm腕相机局部+X，当前状态fresh RGB，Wan和StereoTok左图同源、仅后者多右图。OpenWAM客户端发送right_images，两眼结构性空槽保持黑色，不放宽真实相机右图必需检查；fresh图像不再额外rotate180。LingBot客户端显式--stereo，fresh图不再额外垂直翻转，保持原生动作、settle、步数、guidance和反馈节奏。

新增OpenWAM双目传输/结构空槽回归及XPolicyLab真实渲染验收：Wan/Stereo左图exact、右图非复制、相机外参6/2cm、render物理state不变。Windows AST及diff检查通过；本机无pytest，未冒称单元测试通过；真实环境测试随精确提交部署后执行。运行目录和结果见StereoTok/docs/26-10-22/26-10-22-libero-six-eval-preflight.md。当前未声称GPU验证或抽样通过。
