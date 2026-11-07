# 本机验证记录（2026-10-03）

全部在新建独立 venv 中执行。原 zip、两份 MJCF 与 19 STL 未改动。

| 检查 | 实际结果 |
|---|---|
| 原包零动作 smoke | 两模型各 500 步，均无提前终止或 MuJoCo 警告 |
| 非 RL 模型检查 | 6 腿、18 可控关节，质量 9.022 kg |
| 站立及逐关节动作 | 1580 控制步 / 15800 物理步；18 个正向小偏移全部响应 |
| 机身高度范围 | 0.133108 至 0.154195 m |
| 最低 upright | 0.99993917 |
| 验模警告/关节超界 | 0 / 0.0 rad |
| 随机动作 | 10 完整 episode，共 6000 步，shape obs=63 / action=18 |
| 随机 reward 范围 | -0.39111596 至 0.23846021 |
| 随机 torque / 越界 | 峰值 7.0 N·m / 0.0 rad |
| 随机终止 | 10 次正常超时；无摔倒、NaN 或 MuJoCo 警告 |
| Gymnasium checker | 通过；无界 Box 范围提示见 README |
| 边界回归 | 8/8 通过（seed、非法 action、clip、速度坐标、姿态/高度、关节、timeout、failure/reset） |
| PPO 短训练 | 0 → 4096 步；16 次有限 loss/参数/优化器检查 |
| PPO 恢复 | 4096 → 5120 步；4 次更新检查 |
| 保存/加载一致性 | 恢复后计数及更新次数增加，策略参数确实改变；同 checkpoint 两次载入输出完全一致 |
| 无界面推理 | 2 × 600 步通过 |
| 图形窗口推理 | 1 × 600 步通过，窗口正常关闭 |
| 独立环境依赖 | pip check: No broken requirements found |

## 推理的真实运动结果

- Episode 0: X 位移 -0.000145474 m，平均速度 -0.000012123 m/s，reward -0.012882895。
- Episode 1: X 位移 0.000104735 m，平均速度 0.000008728 m/s，reward -0.004325706。

**这代表训练/保存/恢复/推理闭环通过，不代表学会行走。当前确定性策略基本原地站立。**

## 证据位置

- logs/inspect_robot.json、logs/robot_standing.png
- logs/random_test.json
- logs/smoke/training_summary.json、progress.csv、0.monitor.csv、TensorBoard 事件
- logs/resume_smoke/training_summary.json
- logs/play.json、logs/play_viewer.json
- checkpoints/smoke/final.zip、checkpoints/resume_smoke/final.zip

训练和推理只导入 SimRobot；实际检查 sys.modules 未载入 dual_side_mujoco_driver。

## 本次文件变更

修改原文件：README.md、mujoco/hexapod_rl_env.py、mujoco/train_ppo.py。
新增：configs/robot.yaml、configs/train.yaml、mujoco/robot_interface.py、sim_robot.py、rewards.py、validation.py；scripts 下的五个入口/引导文件；tests/test_env.py；requirements.txt、requirements-lock.txt、.gitignore；docs、logs、checkpoints 及 .venv。
原模型、mesh、兼容驱动、模型生成器、Linux 原始脚本/依赖文件均保留。
