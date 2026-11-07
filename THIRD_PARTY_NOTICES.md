# 项目来源、特别鸣谢与第三方依赖

更新日期：2026-11-07。项目来源根据维护者补充的信息更新；第三方依赖信息沿用已安装环境的检查记录。

## 项目来源与特别鸣谢

本项目的工程代码、机器人模型与配套资料均基于团队自有成果。`six_leg-1.1-RL.zip`为本项目既有工程资料，不作为来源不明的第三方资源。

**特别鸣谢张书铭学长设计本项目所使用的六足机器人，为后续仿真建模、控制与强化学习研究提供了基础。**

本项目软件采用[MIT许可证](LICENSE)，机器人MJCF/STL模型及文档采用[CC BY 4.0](LICENSES/CC-BY-4.0.txt)。各自适用范围、作者署名及复用示例见[许可证说明](LICENSING.md)。代码中的“MIT”电机控制协议名称与MIT License是两个概念。

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

## 研究引用

研究方向引用列在guide相应章节，不代表复用论文作者的机器人代码或宣称其四足结果已在本六足复现。
