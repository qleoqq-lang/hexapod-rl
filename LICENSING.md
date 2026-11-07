# 许可证适用范围

本仓库按文件类型分别授权：软件采用MIT，机器人模型、网格及文档采用CC BY 4.0。它们各自适用于下表范围，并非每个文件都可在两种许可证中任选一种。

## 文件范围

| 范围 | 许可证 |
|---|---|
| `mujoco/*.py`、`mujoco/*.sh`、`scripts/`、`tests/`中的程序代码 | [MIT](LICENSE) |
| `configs/`中的YAML、`requirements*.txt`、`mujoco/requirements-linux.txt`、`.gitignore`及其他自有软件配置/构建文件 | [MIT](LICENSE) |
| `mujoco/hexapod_rl.xml`、`mujoco/hexapod_dynamic.xml` | [CC BY 4.0](LICENSES/CC-BY-4.0.txt) |
| `meshes/`中的全部STL模型 | [CC BY 4.0](LICENSES/CC-BY-4.0.txt) |
| `README.md`、`THIRD_PARTY_NOTICES.md`、本说明、`guide/`和`docs/`中的自有文字、图表和数据资料 | [CC BY 4.0](LICENSES/CC-BY-4.0.txt) |

文档中的自有程序代码片段采用MIT，以便直接复制到程序中。未另行标注且不属于上述CC BY 4.0范围的自有软件文件，默认采用MIT。许可证原文本身保留原文本，不另行改写授权；第三方依赖、引用材料及其已有许可声明优先遵循各自条款。

MIT完整文本见根目录[LICENSE](LICENSE)。CC BY 4.0完整文本见[LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)，官方页面为<https://creativecommons.org/licenses/by/4.0/>。

## 作者与署名

项目贡献者以“hexapod-rl contributors”作为集体署名，各贡献者保留各自贡献的版权；该署名不表示将全部成果的权利转让给仓库维护者。

**六足机器人设计：张书铭。特别鸣谢张书铭学长为本项目的仿真建模、运动控制与强化学习研究提供机器人设计基础。** 软件开发、仿真整理和教学文档由项目贡献者完成。

分享模型或文档时，可使用下述署名，并附上相关许可证和修改说明：

> 来源：hexapod-rl（https://github.com/qleoqq-lang/hexapod-rl）；六足机器人设计：张书铭；仿真整理及文档：hexapod-rl contributors。模型与文档采用CC BY 4.0（https://creativecommons.org/licenses/by/4.0/）。修改情况：请如实注明未修改或列出主要修改。

署名示例不额外增加许可证条件，也不表示原作者认可或担保衍生项目。MIT软件再分发应保留原版权声明和许可证；CC BY 4.0材料分享应保留适当署名及已有声明、附许可证链接并说明修改。具体义务以相应许可证完整文本为准。

第三方运行依赖的说明见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
