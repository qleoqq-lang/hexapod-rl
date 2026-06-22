# 🕷️ hexapod-rl — 六足机器人强化学习步态优化与路径规划

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org)
[![PyBullet](https://img.shields.io/badge/PyBullet-3.25-orange?style=flat)](https://pybullet.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **RL-based gait optimization and path planning for hexapod robots.**
> 使用 PPO/SAC 强化学习算法优化六足机器人步态，结合路径规划实现复杂地形自主导航。

---

## 📋 大创项目概述

本项目是**大创项目（六足机器人）**的强化学习部分，负责：

| 模块 | 方法 | 目标 |
|:----:|:----:|:----:|
| **步态优化** | PPO / SAC | 在平地、斜坡、崎岖地形上学习高效稳定步态 |
| **路径规划** | A* / RRT / RL | 在障碍物环境中规划最优路径 |
| **仿真平台** | PyBullet + URDF | 高保真物理仿真 |

---

## 🏗️ 项目结构

```
hexapod-rl/
├── sim/                  # 仿真环境
│   ├── hexapod.py        # 六足机器人 URDF 模型 + 运动学
│   ├── config.py         # 仿真配置参数
│   └── terrain.py        # 地形生成（平地/斜坡/崎岖）
├── rl/                   # 强化学习算法
│   ├── ppo.py            # PPO 实现
│   ├── sac.py            # SAC 实现
│   ├── buffer.py         # 经验回放缓冲区
│   └── reward.py         # 奖励函数设计
├── gait/                 # 步态规划
│   ├── gait_planner.py   # 步态模式生成器（三角步态/波纹步态/波动步态）
│   └── inverse_kinematics.py  # 单腿逆运动学
├── planner/              # 路径规划
│   ├── path_planner.py   # A* / RRT 路径规划
│   └── navigation.py     # 导航控制器
├── train.py              # 训练入口
├── eval.py               # 评估脚本
├── visualize.py          # 训练可视化
├── requirements.txt      # 依赖清单
└── README.md             # 本文档
```

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/qleoqq-lang/hexapod-rl.git
cd hexapod-rl

# 安装依赖
pip install -r requirements.txt
```

### 训练步态

```bash
# 使用 PPO 训练三角步态（默认）
python train.py --algo ppo --gait tripod --timesteps 5_000_000

# 使用 SAC 训练
python train.py --algo sac --gait ripple --timesteps 3_000_000

# 在崎岖地形上训练
python train.py --algo ppo --terrain rough
```

### 评估与可视化

```bash
# 加载已训练模型并评估
python eval.py --checkpoint runs/ppo_tripod/best_model.zip

# 可视化训练曲线
python visualize.py --logdir runs/
```

### 路径规划

```bash
# A* 路径规划演示
python planner/path_planner.py --algo astar

# RRT 路径规划
python planner/path_planner.py --algo rrt
```

---

## 🧠 强化学习设计

### 状态空间 (Observation Space)

```
[机身姿态 (roll, pitch, yaw) × 3]
[机身角速度 × 3]
[机身线速度 × 3]
[各足关节角度 × 18]      # 6条腿 × 3个关节
[各足足端触地状态 × 6]
[地形高度采样 × 9]
```

### 动作空间 (Action Space)

```
[各足关节目标角度 × 18]   # 连续动作空间 [-1, 1]
```

### 奖励函数

| 奖励项 | 权重 | 说明 |
|:------:|:----:|:------|
| 前进速度 | +1.0 | 鼓励向前移动 |
| 能量效率 | -0.1 | 惩罚关节能耗 |
| 机身稳定性 | -0.5 | 惩罚机身剧烈晃动 |
| 触地奖励 | +0.2 | 鼓励足端有效触地 |
| 存活奖励 | +0.1 | 不倒即奖励 |
| 目标接近 | +2.0 | 接近目标位置的增量奖励 |

---

## 📈 实验结果

> 🚧 训练中，结果将在 baseline 完成后更新

| 算法 | 地形 | 平均步速 | 成功率 | 能耗 |
|:----:|:----:|:--------:|:------:|:---:|
| PPO | 平地 | TBD | TBD | TBD |
| PPO | 斜坡 | TBD | TBD | TBD |
| SAC | 崎岖 | TBD | TBD | TBD |

---

## 🗺️ 路线图

- [x] PyBullet 仿真环境搭建
- [x] 六足机器人 URDF 模型
- [x] PPO 算法实现
- [ ] SAC 算法实现
- [ ] 三角步态训练 baseline
- [ ] 波纹步态训练
- [ ] 崎岖地形适应
- [ ] A*/RRT 路径规划集成
- [ ] 仿真到实物迁移 (Sim-to-Real)

---

## 📚 参考文献

- [PPO: Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [SAC: Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905)
- [Legged Robots that Balance (Raibert, 1986)](https://mitpress.mit.edu/9780262633521/)
- [敏捷机器人步态: MIT Cheetah 系列](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7084369)

---

## 👤 作者

**Q_Leo** — Soochow University AI 2025

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE)
