# 02 机器人模型与环境生命周期

源码：[hexapod_rl.xml](../mujoco/hexapod_rl.xml)、[hexapod_rl_env.py](../mujoco/hexapod_rl_env.py)。动作控制与状态坐标见 [03](03_observation_action.md)。

## 模型事实

默认 hexapod_rl.xml 使用 mesh 视觉、box/capsule 碰撞，引用同级 meshes/ 的 19 个 STL。hexapod_dynamic.xml 保留作比较，不能无验证互换。

| 内容 | 当前值/含义 |
|---|---|
| 结构 | 自由基座 + 六条腿，每腿 coxa/femur/tibia 三个 hinge |
| 质量 | 合计 9.022 kg，来源为原模型，未实机标定 |
| nq / nv / nu | 25 / 24 / 18 |
| qpos (25,) | 基座位置3、四元数4、关节角18 |
| qvel (24,) | 基座速度6、关节速度18；不直接作为机身观测 |
| ctrl (18,) | 当前 unit-gear motor 的力矩指令 |
| 关节范围 | coxa ±1.0472、femur ±1.5708、tibia ±2.3562 rad |
| 执行器 | 18 motor，gear=1，ctrlrange ±7 N·m |
| 物理 | timestep=0.002 s，implicitfast，Newton，iterations=50 |
| 关节参数 | damping=0.05、armature=0.001，来自模型 |
| 地面 | 单平面，friction=[1.0,0.01,0.001]；当前 condim=3 |
| 碰撞 | 机器人与地面交互，未启用腿间自碰撞 |

MjModel 保存结构参数，MjData 保存这份仿真的位置、速度、控制和接触。SimRobot 构造时按名称检查关节/执行器、地址和传动比，然后恢复 stand。

一个环境 step 固定目标并执行 frame_skip=10 个物理步：控制 dt=0.02 s、策略频率50 Hz，PD 按500 Hz更新。render 只显示，不推进物理；mj_forward 刷新派生量，不增加仿真时间。

原模型 lm_coxa capsule 与网格主轴可能不符；生成器按 X 轴构造 capsule，不能逐字复现当前 XML。生成器默认会覆盖训练 XML，应使用候选输出路径重新验模。详见 [原始资产审计](../docs/archive_analysis.md)。

## Gymnasium 接口

HexapodMuJoCoEnv(gym.Env) 的构造读取配置，创建 SimRobot，声明 Box 空间，建立 max_steps 和回合记忆。基类提供协议与随机数支持，不自动实现机器人。

```python
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step(action)
```

单环境 obs=(63,) float32、action=(18,)，reward 为 float，两个结束标记为 bool，info 为 dict。spaces.Box 是格式声明，不自动缩放数据。info 的诊断不会自动进入策略输入。

## reset 的两层职责

```text
Env.reset
  → super().reset(seed)：准备环境随机数
  → SimRobot.reset：mj_resetDataKeyframe(stand)
      → 可选初态扰动 → 关节限位 → ctrl 清零
      → mj_forward → 健康检查
  → last_action 清零、step_count=0、needs_reset=False
  → 返回初始 obs 和诊断
```

stand 基座 z=0.13 m、单位四元数，命令关节姿态约每腿 [0,-35,110] 度。接触后站立高度不必等于 keyframe 高度。

reset 噪声：XY 均匀 ±0.005 m、关节均匀 ±0.015 rad、qvel 高斯标准差0.01（对应平移或转动速度单位）。options={"noise": False} 只取消本次初态扰动。相同 seed 可重复初态；自动 reset 通常沿随机序列继续，而非每回合强制同一个 seed。

构造 SimRobot 的初始化不能代替 Env.reset：环境开始时 needs_reset=True。reset 不清除已学到的网络参数。

## step 的时间顺序

```text
检查 needs_reset
 → apply_action：推进物理，获得 telemetry/applied_action
 → get_state：执行后状态
 → termination_reasons → terminated
 → step_count+=1 → truncated
 → compute_reward（本次 applied、旧 last_action）
 → 检查 reward 有限
 → last_action 更新为本次 applied
 → needs_reset = terminated or truncated
 → 返回新 obs、reward、标记及 info
```

这保证 reward 比较相邻两步动作，而新观测的动作历史是刚采用的动作。提前更新 last_action 会让动作平滑差值为零。环境不会在 step 内偷偷 reset。

info 包括高度、世界 X 速度、upright、小腿接触数，以及 energy、peak_torque、最大关节越界、reward_terms 和 termination_reasons。applied_action 被取出用于历史，不留在最终 info 中。

## 回合结束与异常

upright=-projected_gravity[2]，直立约1，横躺约0，倒置约-1。

| 条件（严格按源码比较） | 结果 |
|---|---|
| upright≤0 | flipped |
| 否则 upright<0.4 | excessive_tilt；0.4约对应总倾斜66.4度 |
| base z<0.10 m / >0.30 m | height_low / height_high |
| 控制步内实际关节最大越界>0.02 rad | joint_limit |
| step_count≥int(12/0.02)=600 | truncated=True，不添加失败原因 |

有失败原因就 terminated=True。高度、姿态、关节原因可以叠加；失败与超时同一步发生时两个标记都为 True。0.30 m 与0.02 rad 是实验保护阈值，不是厂家实机限制。

NaN/Inf 动作、错误 shape 在物理推进前报错；状态非有限或 MuJoCo warning 抛异常停止，属于数值故障路径，不是普通失败回合。Box 为无穷边界，contains 不能替代完整的有限性检查。

## 训练和推理的 reset 边界

| 层 | reset 返回 | step 返回 | 结束后谁 reset |
|---|---|---|---|
| 原始 Env | obs, info | 五元组 | play/random_test 外层循环 |
| SB3 VecEnv | 批量 obs | obs, rewards, dones, infos | DummyVecEnv |

VecEnv 保存旧最后观测到 infos[i]["terminal_observation"]，再将该环境新回合初始观测放回 obs 对应行。done=terminated or truncated；TimeLimit.truncated=truncated and not terminated。普通超时价值补偿由 SB3 处理，见 [05](05_ppo.md)。

自查：第600步同时 height_low，两标记均为 True；随后直接调用原始 Env.step 会被 needs_reset 拒绝。
