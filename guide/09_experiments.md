# 09 训练、日志与可复现实验

本页是操作命令的统一入口，均从工程根目录执行，适用于已验证的Windows PowerShell/CPU环境。算法机制见 [05](05_ppo.md)，修改定位见 [06](06_modification_guide.md)。下列正式训练命令是操作方案，不表示本次已经执行了长训练。

## 环境准备

现有.venv可直接使用；不需要激活或覆盖全局环境。新机器应选择Python 3.11建立独立环境，先确认版本，再安装：

```powershell
cd D:\1hexapod_robot\six_leg_lzx-1.1-RL
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

只在尚无目标venv的新机器执行创建命令。requirements.txt固定直接依赖，requirements-lock.txt是已有环境的完整快照；可选择后者安装。它们不是跨平台、带wheel哈希的供应链锁，不能保证所有平台均有对应wheel。PyTorch设备以实际torch版本与cuda.is_available()为准，不从requirements推断一定是CPU或CUDA。当前实测为CPU；Linux旧requirements仅作来源参考，尚未完整验证Linux安装。

## 第一次正式PPO训练前清单

- 确认能说明63维观测和18维位置偏移的单位、顺序、坐标；读完00～05主线。
- 保存当前代码/配置/依赖版本与种子。本目录当前未初始化独立Git仓库，可先保留SHA256清单；发布并建立Git后记录commit及未提交diff。
- 核对模型碰撞和参数的已知限制，避免把仿真值当成实机参数。
- 按下列顺序通过验模、随机和边界测试；默认报告必须与当前契约一致。
- 做一次短训练、恢复与推理，确认文件实际产生且数值有限。
- 先确定正式预算、评估种子、对照及指标；选择全新run-name，检查磁盘空间，保存失败结果。
- 当前自动零动作对照尚缺，应先补一份最小评估记录或通过既有Env接口手工实验；没有基线时不宣称“训练使行走改善”。

```powershell
.\.venv\Scripts\python.exe scripts\inspect_robot.py
.\.venv\Scripts\python.exe scripts\random_test.py --episodes 10
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\train.py --steps 4096 --run-name first_smoke_01
.\.venv\Scripts\python.exe scripts\play.py --checkpoint checkpoints\first_smoke_01\final.zip --headless --episodes 5 --seed 100 --output logs\first_smoke_01\play_seed100.json
```

前两个默认报告会被覆盖，已有重要记录先另存。可用--output留档，但训练入口仍只读默认文件名。run-name重复会报错，重复练习需改名。GUI验模可加--render；推理去掉--headless即可，模型生成器不属于日常启动步骤。

## 开始基线、改训练参数与恢复

准备完成后，可从头训练默认20000步预算：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --steps 20000 --run-name forward_baseline_01
```

20000只是起始实验预算，不能保证学会走路；默认rollout下实际为20224步。若要改变PPO超参数，将configs/train.yaml复制为一个实验配置，只改一个变量，并使用：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --train-config configs\train_experiment.yaml --steps 20000 --run-name forward_variant_01
```

上例train_experiment.yaml需由你按说明先创建，仓库不附一份重复默认配置。改变训练seed也在这份YAML中，入口没有--seed选项。

恢复默认基线，追加交互预算：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --resume checkpoints\forward_baseline_01\final.zip --steps 20000 --run-name forward_resume_01
```

恢复遵循 [01](01_training_flow.md) 的参数规则，不能靠新YAML覆盖旧learning_rate/n_steps/n_epochs。记录初始/结束计数，恢复不精确重放仿真现场。

## 监控与日志阅读

```powershell
.\.venv\Scripts\tensorboard.exe --logdir logs
```

| 文件/字段 | 如何解释 |
|---|---|
| <环境编号>.monitor.csv 的r/l/t | 原始回合总reward、回合控制步数、记录时间；首次episode结束前可没有数据行 |
| rollout/ep_rew_mean、ep_len_mean | 最近回合窗口统计，不是独立测试结果 |
| robot/forward_velocity、upright、base_height | 日志汇总窗口的状态均值，不代表最差姿态或失败率 |
| reward/* | 六项已加权评分均值；robot/energy是物理量 |
| train/value_loss | Critic对return目标误差；尺度受reward影响 |
| train/explained_variance | 旧价值对目标的解释程度，接近1较好；目标方差为0时可为NaN |
| train/approx_kl、clip_fraction | 策略变化和目标裁剪活跃程度；结合行为看，不设通用万能阈值 |
| train/std、entropy_loss | 动作分布的探索尺度；不等于实际关节运动幅度 |
| train/loss | 本地SB3最后一个batch的总loss，不是整个rollout平均 |
| train/n_updates | 每epoch增加，不等于optimizer.step次数；本地该键不写入TensorBoard，可查CSV |
| training_summary.json | 正常收尾的有限性检查、实际计步和保存路径 |

SB3在采样后、更新前dump一轮日志，train指标常来自前一次更新；项目正常结束另dump最后指标。先按实际计步对齐，不要求所有列每行都有数值。FiniteTraining将所有记录到的非有限train标量视为异常，常数目标导致的explained_variance NaN也可能被中止；这是待评估的检查策略，不自动断言网络已数值崩溃。

收敛没有单一loss阈值。用固定测试集看速度/失败是否持续改善，多个训练种子是否一致；停滞需先看分项、动作探索、饱和和初态。策略退化要比较多个保存点，不只看final或最好的一条视频。

## 独立评估与对照

```powershell
.\.venv\Scripts\python.exe scripts\play.py --checkpoint checkpoints\forward_baseline_01\final.zip --headless --episodes 10 --seed 1000 --output logs\forward_baseline_01\evaluation_seed1000.json
```

这是当前已存在的推理评估，输出逐回合净位移、速度、总reward、时长和原因，不自动算成功率、挑最佳模型或比较零动作。固定相同配置、reset种子集合、时限和确定性模式。训练中自动评估、独立验证集和测试集调度尚未实现，先用实验记录人工管理。

建议协议：预先登记训练seed集合、用于选checkpoint的验证seed、只在最终汇报使用的测试seed；随机数种子不同本身不能保证地形分布独立。当前只有同一平面，不能把新seed称为未知地形测试。

零动作对照通过env.reset(seed)、连续env.step(np.zeros(18))实现；策略对照用model.predict。二者必须各自从同一初态重置，按原始终止规则记录；零动作仍有PD。当前smoke_test固定seed=7且只汇总有限字段，不是现成的公平成套对照工具。

每回合先记实际计数，再算均值/分散程度；有失败回合也纳入。报告任务失败数/总回合数，若要跌倒率先定义原因集合。速度跟踪RMSE需要目标指令，当前任务没有，不能拿0.25 cap充当目标。仅增加reward不等于能力提升。

## 实验记录规范

复制 [实验模板](../docs/experiment_template.md) 到本地logs/<run>/。至少保留：假设与唯一变量、完整命令、训练与评估seed、有效模型参数、代码版本/哈希、依赖/设备、任务契约、请求/实际预算、checkpoint来源及哈希、逐回合指标、结论和局限。

当前程序自动保存部分配置/契约，但没有自动保存完整代码版本、命令、评估seed和恢复后的全部有效参数。模板补齐这些缺口；失败、未收敛、无效对照也应记录。不要事后只挑有利seed或反复用测试集调参。
