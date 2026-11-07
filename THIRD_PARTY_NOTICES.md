# 资源来源、依赖许可与发布准备

检查日期：2026-11-05。本文件记录已查到的归属和缺口，不授予原本不存在的授权，也不替项目作者选择许可证。

## 工程与机器人资产

| 资源 | 已知来源/证据 | 当前状态 |
|---|---|---|
| 原Python驱动、生成器、环境和训练基础 | six_leg-1.1-RL.zip历史文件清单；docs/README_original.md | 原项目署名、上游地址和许可证未明确 |
| 两份MJCF、19个STL、质量/惯量与关节数据 | docs/archive_manifest.json、archive_analysis.md、mesh_audit.json | 模型/网格作者和公开再分发授权未确认 |
| 后续拆分接口、验证与教学文档 | 工作区现有修改，历史验证记录与本次审查报告 | 需由实际权利人确认可授权范围与署名 |

当前工程及原包清单未发现LICENSE/COPYING/NOTICE授权文件。工作区此次未定位到原zip实物，使用留存manifest与历史分析核查，不能据此重新认证原包来源。应由持有人补充上游仓库/作者、获取方式及许可文本。

“MIT”在此工程部分注释中指电机控制接口语义，不能据此认定原资源使用MIT License。未添加一个笼统LICENSE覆盖源码、模型和mesh，也未把未知作者归属改为当前维护者。

## 运行依赖

以下来自本地已安装distribution元数据；真正分发wheel/二进制时还需保留包内LICENSE/NOTICE及所含第三方条款。依赖的许可证不会自动给本项目资产授权。

| 依赖 | 本地元数据标记 |
|---|---|
| MuJoCo | Apache-2.0 |
| Gymnasium、Stable-Baselines3、PyYAML | MIT |
| TensorBoard | Apache 2.0 |
| Pillow | MIT-CMU |
| NumPy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| PyTorch | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |

NumPy/PyTorch包含多种第三方组件，不能把整个已安装包简单归为单一许可。建议源码仓库通过requirements安装依赖，不直接上传.venv；原始Linux参考依赖与当前主依赖的差异见审查报告。

## 公开发布前逐项确认

- 确认原工程、两份MJCF和19个STL的作者、来源与公开分发权限；保留可核对证据。
- 由权利人选择自有代码/文档的LICENSE，明确第三方模型与网格是否另用许可或暂不分发。
- 若不能公开mesh，准备合法获取说明及资产完整性验证；当前模型启动依赖这些资源，不能删掉后宣称完整可运行。
- 检查实际暂存清单；.gitignore只影响未跟踪文件，不会移除已经跟踪的日志或权重。
- 确认无个人路径/敏感凭据、缓存、虚拟环境、原zip或大型训练产物被暂存。不要直接复制全部工作区。
- 在独立新环境验证安装/启动，并记录平台与失败限制；本次未联网重装依赖。
- 通过后再初始化/配置独立仓库、提交并审阅暂存diff。本次没有初始化仓库、提交或推送。

研究方向引用列在guide相应章节，不代表复用论文作者的机器人代码或宣称其四足结果已在本六足复现。
