# 03 观测、动作与关节控制

源码：[robot_interface.py](../mujoco/robot_interface.py)、[sim_robot.py](../mujoco/sim_robot.py)。本章是输入输出含义的统一参考，修改网络接口前先查这里。

## 关节顺序、符号与状态地址

LEG_ORDER=[FL,ML,RL,FR,MR,RR]，对应模型前缀 [lf,lm,lr,rf,rm,rr]。每腿依次 coxa、femur、tibia，组成18维向量。

| 关节向量下标 | 腿 | 示例关节名 |
|---|---|---|
| 0～2 | FL | lf_coxa_joint、lf_femur_joint、lf_tibia_joint |
| 3～5 | ML | lm_*_joint |
| 6～8 | RL | lr_*_joint |
| 9～11 | FR | rf_*_joint |
| 12～14 | MR | rm_*_joint |
| 15～17 | RR | rr_*_joint |

通常命令到 MJCF 的 signs=[-1,+1,-1]，FR 为[-1,-1,-1]。因为每个 sign 的平方为1，同一映射可来回转换角度和速度。

名字 → joint_ids → jnt_qposadr/qadr、jnt_dofadr/vadr；执行器另按 joint名+"_motor" 查 actuator_ids。joint ID 不等于 qpos 下标：自由基座占7个 qpos、6个 qvel，必须查询地址。

## 状态来源

SimRobot.get_state() 从 MuJoCo 取状态，创建 RobotState。@dataclass 生成常用构造方法；类型注解不检查维度、单位或自动复制数组。

| 状态 | shape | 来源/含义 |
|---|---|---|
| base_position | (3,) | xpos：base_link 原点的世界位置，m |
| base_quaternion | (4,) | xquat：wxyz 朝向 |
| linear_velocity | (3,) | body质心线速度，转为base_link轴，m/s |
| angular_velocity | (3,) | base_link轴角速度，rad/s |
| world_linear_velocity | (3,) | 同一质心的世界轴线速度，供reward/诊断 |
| projected_gravity | (3,) | 世界单位重力在机身轴的方向 |
| joint_position / velocity | 各(18,) | qpos/qvel 经地址选择、乘signs，rad / rad/s |
| tibia_contacts | (6,) | 小腿几何体接地诊断，不是实机足端传感器 |

mj_objectVelocity(..., 0) 返回世界轴速度，顺序为角3、线3；代码通过 rotation=data.xmat[base_id].reshape(3,3)，用 rotation.T 转成机身轴。不能直接把 qvel[:6] 当成这种观测，也不能把旋转过的惯性轴当 base_link 轴。

base_position 与线速度的参考点分别为 body 原点、质心；精确比较位移导数与速度时应统一参考点。get_state 不推进物理。

## Observation 的63个数

调用链：Env.get_observation → robot.get_observation(last_action) → get_state → make_observation(state, nominal_command, previous_action)。

| Python切片 | 内容 | shape | 单位/处理 | 策略用途 |
|---|---|---|---|---|
| 0:3 | linear_velocity | (3,) | m/s，机身轴，无额外缩放 | 当前移动状态 |
| 3:6 | angular_velocity | (3,) | rad/s，机身轴 | 当前摇晃/转动 |
| 6:9 | projected_gravity | (3,) | rotation.T@[0,0,-1]，单位方向 | 倾斜程度 |
| 9:27 | joint_position-nominal | (18,) | rad，命令方向，相对站姿 | 关节位置偏差 |
| 27:45 | joint_velocity | (18,) | rad/s，命令方向 | 关节运动趋势 |
| 45:63 | previous_action | (18,) | 最终采用的目标偏移比例 | 上一控制目标 |

np.concatenate 首尾拼接：3+3+3+18+18+18=63；astype(float32) 只统一类型。没有 VecNormalize，也没有按组标准化；位置减 nominal 是中心化，仍为弧度，未除 action_scale。Box 声明不会归一化。

base高度、四元数、接触和世界速度不直接拼入策略。重力能表示倾斜，不能区分yaw；没有目标速度或航向命令。因此当前观测不能直接支持准确的命令跟踪。实机需要估计机身速度，不能假设装了速度计。

教学例：机身速度[0.2,0,0]、角速度[0,0,0.05]、重力[0,0,-1]；仅首关节位置偏差0.1、速度0.3、上一动作0.5，则非零项为 obs[0]=0.2、[5]=0.05、[8]=-1、[9]=0.1、[27]=0.3、[45]=0.5。理想站姿也不是全零观测。

## Action 从网络到关节

```text
Actor → 原始动作 → SB3 clip [-1,1]
 → 单环境 action (18,)
 → SimRobot 检查有限性、shape，再 clip
 → nominal_command + action_scale×action
 → signs 转为 MJCF 角度 → 关节目标限位
 → 每个物理子步计算 PD → 力矩限幅 → ctrl → mj_step
```

action 是相对 stand 的目标位置偏移比例，不是相对上一关节角的累积增量。相同 action 连续发送，目标不累加。零 action 请求 stand，仍有保持姿态的力矩。

控制核心：
```python
target = clip((nominal_command + 0.35 * action) * signs,
              joint_lower, joint_upper)
torque = clip(kp * (target - q) - kd * v, -7, 7)
```

kp、kd、q、v、target、torque 均为(18,)。kp=20 N·m/rad；kd每腿[0.5,0.8,0.8] N·m·s/rad。标量乘向量是广播，signs 相乘是逐元素操作，limits[:,0/1]分别取18个下/上界。

每个控制步目标不变，10个物理子步重新读q/v并计算力矩。目标位置不会直接写入 qpos，实际运动由动力学决定；只写 ctrl 而不 mj_step 不会推进仿真。

教学算例：第0关节 lf_coxa，nominal=0、sign=-1；action=0.5 → target=-0.175 rad。若q=v=0，初始力矩=20×(-0.175)=-3.5 N·m，随后每子步反馈调整。它不表示关节立刻到达目标。

## applied_action 与动作历史

控制层反算 applied_action=(target×signs-nominal)/action_scale，表示限位后接受的目标指令，不是实测运动量。返回 telemetry 还含绝对机械功、力矩峰值、最大实际越界量。

```text
obs_t 的历史块是 applied_(t-1)
 → policy 输出 a_t → apply_action 得到 applied_t
 → reward 比较 applied_t 与旧 last_action
 → last_action=applied_t
 → obs_(t+1) 的历史块是 applied_t
```

PPO buffer 保存原始采样动作及其旧 log_prob；环境收到裁剪动作，历史保存 applied。三者可能不同。上一动作块不是循环网络记忆。

## 接口与修改边界

RobotInterface(Protocol) 只声明 nominal_command、get_state/get_observation/apply_action/close；没有 reset，也不是后端工厂。SimRobot 不必显式继承它；Env 当前直接构造 SimRobot，尚无 RealRobot 实现。

增加/重排观测需同步状态来源、make_observation、observation_space、观测版本、检查脚本与策略兼容性。新增 RobotState 字段不会自动增加网络输入。改变顺序或尺度，即使shape不变，也改变语义；见 [修改索引](06_modification_guide.md)。

自查：手动把 play 的obs加成(1,63)会得到(1,18)动作；SimRobot只接受(18,)，因此应让 predict 自动处理单样本维。
