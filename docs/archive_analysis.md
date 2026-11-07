# 原压缩包分析（2026-10-03）

已解压到工作区的 `six_leg_lzx-1.1-RL/`，保留原 zip。包内共 31 个条目，其中 28 个文件、3 个目录，解压文件合计 100,646,753 字节。所有文件的原始大小和 SHA256 在 `archive_manifest.json`。19 个二进制 STL 均读取并检查边界，结果在 `mesh_audit.json`。

## 原始目录与可复用资源

```text
six_leg_lzx-1.1-RL/
├── README.md
├── meshes/                     # 19 STL: base_link + 六腿各三节
└── mujoco/
    ├── hexapod_dynamic.xml
    ├── hexapod_rl.xml
    ├── create_rl_mjcf.py
    ├── dual_side_mujoco_driver.py
    ├── hexapod_rl_env.py
    ├── train_ppo.py
    ├── requirements-linux.txt
    └── linux_smoke_test.sh
```

| 内容 | 文件/实际发现 | 本次处理 |
|---|---|---|
| 几何外观 | 19 个 STL；lf/lm/lr/rf/rm/rr 各含 coxa/femur/tibia | 全部原样复用 |
| 动力学模型 | `hexapod_dynamic.xml`；自由基座、惯量、质量、18 电机、地面、stand keyframe | 原样保留；原 smoke 实测 500 步通过 |
| 训练模型 | `hexapod_rl.xml`；box/capsule 碰撞、mesh 视觉 | 默认使用；原样保留、没有重造机器人 |
| URDF | 包内没有 | 不臆造、不转换 |
| 关节定义 | 两份 MJCF 的 joint 完全一致；18 hinge + 1 free joint | 按名称严格映射，不按数字猜索引 |
| 电机/舵机参数 | MJCF 的 motor、joint actuatorfrcrange，以及 env/driver 的 PD 参数 | 使用包内值，但不是厂商实机数据表 |
| 控制接口 | `dual_side_mujoco_driver.py` 模拟 MIT 边界，位置/速度/前馈力矩接口 | 复用方向符号与位置 PD 语义；训练不导入此模块 |
| 仿真环境 | `hexapod_rl_env.py`，Gymnasium reset/step/render | 在原结构内拆分控制和 reward，补齐验证 |
| RL | `train_ppo.py`，直接使用 SB3 PPO/MlpPolicy | 沿用同一算法与实现，增加恢复、日志、检查 |
| observation | 原 env 69 维，直接拼 qvel[:6]、重力、关节、接触、上一动作 | 明确机身坐标，改为 63 维；接触仅诊断 |
| action | 原 env 18 维，stand + 0.35 rad × action，MIT PD 后限矩 | 沿用；补目标限位与非法输入检查 |
| reward | 原 env 内 alive/forward/upright/height/energy/action_rate | 去除不一致高度目标，保留简单目标并外置 |
| 配置 | 参数主要硬编码在 XML/Python；Linux requirements、shell 脚本 | 新增集中 YAML 与已验证依赖锁 |
| checkpoint | 无 `.pt/.pth/.ckpt/.zip` 训练权重，无训练日志 | 本次实际短训练生成 |

结论来自源码的 import、类与 XML 内容：**Python + NumPy + MuJoCo + Gymnasium + Stable-Baselines3（PyTorch）PPO**。原入口偏向 Linux（EGL、SubprocVecEnv），不依赖 Isaac Gym 或 PyBullet。第一版在 Windows CPU 下验证。

## 模型和控制事实

- 总质量 9.022 kg；基座 1.906 kg，每腿三节质量分别 0.368/0.379/0.439 kg。
- `nq=25, nv=24, nu=18`；浮动基座姿态四元数占 4 个位置量，不是额外受控关节。
- 顺序 `FL ML RL FR MR RR` ↔ `lf lm lr rf rm rr`，每腿 `coxa femur tibia`。
- 关节范围依次 ±1.0472、±1.5708、±2.3562 rad。轴向和局部坐标并不完全镜像，特别是 rf_femur，不应套用统一四足模板。
- 命令到 MJCF 符号：通常 `[-1,+1,-1]`，FR 是 `[-1,-1,-1]`。原 stand 命令每腿 `[0,-35,110]°`，基座 z=0.13 m。
- 两份 MJCF 的惯量、关节和执行器定义保留。2 ms 物理步，10 倍降采样，50 Hz policy；Kp 每关节 20，Kd 每腿 `[0.5,0.8,0.8]`；±7 N·m 限矩。
- dynamic 模型有 54 个关节标量传感器，加 gyro/accelerometer/framequat；rl 模型仅保留 54 个关节标量传感器。基座状态可直接由 MuJoCo 状态计算，并非假设模型有速度计。

## 已发现的问题和适用范围

1. 原 env 默认 dynamic，但 train 传 rl，两个入口碰撞形状不同；现在统一默认 rl。
2. 原 height reward 目标 0.20 m 与 stand=0.13 m、实测站立高度不一致；本次移除该奖励。
3. 原 `qvel[:6]` 未交代线速度/角速度坐标系；新代码显式统一到 base_link 坐标。实际回归测试发现局部惯性轴与 base_link 轴不同，已修正。
4. 原 smoke 只发零动作；本次先完成逐关节、随机动作和边界测试再训练。
5. 原模型机器人碰撞组只与地面相交，**未启用腿间自碰撞**。粗略 capsule 接触不能当作已校准的真实足端接触。
6. `lm_coxa` 简化 capsule 几乎沿 X 退化，而其网格主要沿 Y 延伸；原生成器也一律按 X 构造 capsule。保留现有模型用于最小闭环，不声称几何已达到 Sim2Real 精度。
7. 原生成器并不能逐字复现当前 rl XML：dynamic 含足端 sphere、IMU，而 rl 的相关结构不同。不要覆盖已验证 XML；如研究生成器，用 `--output` 输出候选文件并重新验模。
8. 原仿真驱动的完整兼容路径要求 `_original_ruckig` 及左右侧电机 XML、方向配置；这些外部资源不在包内。RL 原本用直接 PD streaming，不需要该原生模块，本次保留这种路径。
9. 包内没有实际 SocketCAN/串口通信实现；驱动含硬件接口命名和 CAN ID 配置解析，不能据此把训练接到硬件。本次所有入口只构造 SimRobot，无真实机器人后端或切换参数。
10. 原包没有实机延迟、传感器噪声、摩擦标定、电机热限制、减速比/额定转速数据表。±7 N·m 等只作为源工程仿真参数，不宣称实机已确认。

原 README 字节原样备份为 `README_original.md`。其余修改过的原 Python 可从未改动的 zip 恢复。未改工作区中其他六足/JetHexa 项目。
