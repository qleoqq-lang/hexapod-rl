# 01 训练入口、调度、保存与恢复

源码：[scripts/train.py](../scripts/train.py)、[mujoco/train_ppo.py](../mujoco/train_ppo.py)。操作命令统一放在 [09](09_experiments.md)，本页解释命令如何生效。

## 入口与参数

scripts/train.py 导入 _bootstrap 以准备搜索路径，再调用 train_ppo.main()。__main__ 是 Python 的直接执行入口；导入模块不会自动启动该分支。

main 先读 train.yaml。命令行 --steps、--workers 在提供时覆盖 total_timesteps、workers；--config 选择机器人 YAML，--train-config 选择训练 YAML，--resume 选择已有策略，--run-name 选择结果分组。run-name 必须是普通目录名，已有日志或 checkpoint 目录会拒绝重用。

```text
参数 → require_validation(load_config(robot_config))
     → 创建 logs/checkpoints 新目录
     → make_vec_env(HexapodMuJoCoEnv, n_envs=workers, seed=...)
     → VecCheckNan
     → 新建/恢复 PPO
     → learn → 最后检查 → 保存 final 与摘要 → close
```

require_validation 读取固定的 logs/inspect_robot.json、logs/random_test.json，要求 passed 且 contract 与当前机器人一致，不会自动重跑两个检查。改 robot 配置、XML、mesh 或被哈希源码后，要重新验证。完整契约说明见 [06](06_modification_guide.md)。

## 环境包装和两种模型

make_vec_env 为每个单环境加 Monitor，默认用 DummyVecEnv 在一个进程内顺序推进。workers=2 表示维护两台独立仿真机器人，不表示两个进程。外面再包 VecCheckNan，检查经过边界的观测、动作、奖励数值。

SimRobot.model 是 MuJoCo 的 MjModel；本文件 model 是 SB3 的 PPO 对象。名字一样，职责不同。

## 新建与恢复分支

| 项目 | 新建 PPO | --resume |
|---|---|---|
| 网络和优化器 | 初始化 | PPO.load 恢复 |
| PPO 参数 | 显式传入 YAML 的七项参数 | 使用 checkpoint 中的有效参数 |
| 仿真状态 | 新环境 | 新环境；不恢复物理现场 |
| num_timesteps | 从零开始 | 继承训练计数 |
| learn 的 reset_num_timesteps | True | False |

七项参数是 n_steps、batch_size、n_epochs、learning_rate、gamma、gae_lambda、clip_range。其他 PPO 默认值由当前库决定，见 [05](05_ppo.md)。

恢复分支只显式传 env 和 device，修改 YAML 中 learning_rate 或 n_epochs 后 --resume 不会自动覆盖旧值。当前 cfg 的线程数、worker 数、设备、保存间隔等运行设置仍参与本次执行。worker 数改变后每段经验总量也会改变，不能把这种恢复当作严格连续复现。

当前 robot_config/train_config 副本记录的是读取到的配置，不是恢复模型的完整有效参数清单；正式实验需另记录有效模型属性。任意新增 YAML 键不会自动被 PPO 接收。

## learn 的计步与回合

learn(total_timesteps=steps) 的目标是新增交互步数。没有中断时按完整 rollout 完成，一段有 n_steps×workers 条样本。默认 --steps 1000 实际完成 1024 步；workers=2 时每段有 512 条样本。

回合长度由 robot.yaml 决定，默认 600 控制步。rollout 满了只触发更新，不强制 reset。区分环境步、物理子步、epoch 与 optimizer step，见 [05](05_ppo.md)。

## FiniteTraining 回调

| 方法 | 时机 | 行为 |
|---|---|---|
| check_training | 被下两处调用 | 检查已有 train 指标、policy 参数、optimizer 状态的有限性 |
| _on_rollout_start | 下一段采样前 | 检查前次更新；第一段还没有 train/loss |
| _on_step | 每次向量交互 | 记录 reward/robot 均值，统计结束回合，定期 save |
| _on_training_end | learn 正常收尾 | 再检查最后一次更新 |

_on_step 返回 True 让采样继续。保存条件使用回调调用次数，间隔为 max(1, checkpoint_every // workers) 次向量 step；workers 不整除时不是精确的 checkpoint_every 个样本。

step_*.zip 的文件名是采集计数；回调在采样期间执行，因此刚采的数据未必已经用于更新。没有 EvalCallback，不会自动挑选 best.zip。

## 最终保存与中断边界

learn 返回后必须至少记录一次 train/loss 检查，然后 dump 日志、保存 final.zip，再写摘要、配置副本和契约。finally 关闭环境。发生异常或手动中断时，可能只有之前的周期 checkpoint，没有 final 或完整摘要；不能把遗留文件当作本次成功。

checkpoint 保存网络、优化器、PPO 设置和计数等；不保存 MuJoCo 当前姿态和用于精确重放的全部现场。恢复会开始新回合，不保证等价于从未中断的训练。

| 产物 | 作用 |
|---|---|
| checkpoints/<run>/step_*.zip、final.zip | 周期模型、正常完成的最终模型 |
| logs/<run>/training_summary.json | 请求/实际步数、回合数、设备、有限性检查 |
| train_config.yaml、robot_config.yaml | 本次读取的配置快照 |
| contract.json | 机器人接口/资产一致性记录 |
| *.monitor.csv、progress.csv、TensorBoard 事件 | 回合、训练和诊断日志 |

自查：改 YAML 学习率后恢复模型，应该检查 YAML 副本还是有效模型 learning_rate 与日志？答案是后两者，并追踪 PPO.load 是否传入覆盖值。
