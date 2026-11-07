# 项目证据与整理记录

日常学习从 [guide导航](../guide/README.md) 开始；本目录主要保存来源与可核对证据。

| 文件 | 性质/用途 |
|---|---|
| [documentation_review_20261105.md](documentation_review_20261105.md) | 本次文档重构、代码审查、验证及发布交付记录 |
| [review_snapshot_20261105.json](review_snapshot_20261105.json) | 本次实际环境、源码哈希、测试/训练/恢复/推理摘要，可随源码发布 |
| [experiment_template.md](experiment_template.md) | 后续正式实验的统一记录模板 |
| [archive_analysis.md](archive_analysis.md) | 2026-10-03原包资源分析、复用和历史改动 |
| [archive_manifest.json](archive_manifest.json) | 原始文件大小与SHA256清单 |
| [mesh_audit.json](mesh_audit.json) | 原mesh边界审计 |
| [environment.json](environment.json) | 2026-10-03环境快照，不代表永远的本机状态 |
| [validation_results.md](validation_results.md) | 2026-10-03运行证据及本地原产物位置 |

原README的69维观测、旧输出目录和旧安装命令已过时；当前是63维，操作以guide/09为准。旧Linux依赖/脚本留作来源与兼容性调查，本次没有验证其完整安装流程。历史记录中的“本次”均指该文档日期，不自动描述后来的改动。

logs/checkpoints由.gitignore排除。源码仓库读者可复查这里的摘要，完整原始训练产物需本地生成或由维护者另外提供；不能把未随仓库提供的历史路径当作已附带下载。
