# 06 修改影响与故障排查索引

使用顺序：明确假设 → 找到参数消费处 → 确定兼容性影响 → 保存基线 → 一次一项修改 → 基础检查 → 同协议对照。无需掌握所有数学证明，但必须能说明输入输出、单位、边界与判断证据。

## 修改位置与验证

| 想改什么 | 当前消费位置 | 连带影响/验证 |
|---|---|---|
| 训练步数/环境数 | train_ppo.main → learn / make_vec_env | 实际rollout取整、样本总数；当前非多进程 |
| 学习率、batch、epoch | train.yaml → 新建PPO | --resume使用已存参数；核对有效属性及日志 |
| 前进收益/cap | robot.yaml.reward → compute_reward | cap不是速度命令；契约改变，重验及新训练 |
| 姿态/动作/机械功代价 | robot.yaml.reward → rewards.py | 查加权terms与独立运动指标，不跨reward定义只比总分 |
| 更早结束倾斜回合 | termination.min_upright → termination_reasons | 提高阈值允许的倾斜更小；不是增大失败扣分 |
| episode长度/初态扰动 | Env.__init__/reset → SimRobot.reset | dt取整、历史清零、初态合法；改变任务契约 |
| 动作尺度 | action_scale_rad → apply_action | 目标角、跟踪误差、饱和与限位；零动作仍为stand |
| kp/kd/力矩限制 | apply_action PD、XML motor/关节限制 | 跟踪、稳定、负载；不能超过模型上限或当成实机额定值 |
| frame_skip/物理dt | robot.yaml / MJCF option | 控制频率、PD、episode步数、rollout时长和reward尺度 |
| stand参考姿态 | MJCF keyframe → reset / nominal_command | 高度、接触、动作中心和关节限位都需重验 |
| 增加观测 | get_state → RobotState（如需）→ make_observation → observation_space | 更新版本、shape、测试及训练；仅加状态字段不改变网络 |
| 增加/改变action | 模型/执行器、JOINT_NAMES/signs、SimRobot映射、Env空间、验模/测试 | 当前多个位置硬编码18及25/24/18；不是改一个Box即可；旧模型不兼容 |
| 更换模型 | model路径 → SimRobot.__init__ | 关节名、轴、质量惯量、keyframe、gear及碰撞；路径可加载不代表已适配 |
| 未来地形/域随机化 | 模型构建、reset、接触诊断与评估协议 | 当前无插件机制；先独立候选模型与测试，见10 |

当前接口把状态/动作从仿真中抽离，但Env直接创建SimRobot；无需立即增加后端注册器或新RL框架。需要实机时再明确reset、时钟与故障协议。

## 契约与复现不是同一件事

validation.contract 包括：

- OBSERVATION_VERSION、JOINT_NAMES。
- 整个robot配置，模型路径规整为文件名。
- 当前XML、meshes目录的*.STL、sim_robot.py、robot_interface.py、hexapod_rl_env.py、rewards.py的SHA256。

不包括train.yaml、train_ppo.py、依赖版本、Git版本和所有运行参数。故它是接口/资产一致性门，不是完整实验版本。

即使仅改被哈希源码的注释、换行或reward权重也可能不匹配。重新验模可刷新报告，不会修改旧checkpoint中的契约。不要通过删除检查或伪造元数据把不兼容策略送入环境；如需有意迁移权重，应单独设计并验证语义。

## 不应顺手改掉的顺序

1. compute_reward必须先读旧last_action，再更新历史。
2. Env.step返回旧回合最后状态；训练包装层负责保存它再reset。
3. GAE按时间先计算目标，再打乱mini-batch。
4. 修改reward.termination只改扣分，终止条件在另一处。
5. 物理推进在mj_step，render和sleep都不驱动机器人。

## 按现象查上游

| 现象 | 从哪里追踪 | 首个可验证问题 |
|---|---|---|
| Validation stale / contract mismatch | require_validation、PPO.load后的比较 | 读了哪份报告？哪个配置/文件变了？ |
| 修改学习率恢复后没变 | PPO.load分支 | 新参数是否实际传给模型？ |
| 第一次没有train/loss | rollout → 首次PPO.train | 是否还没完成第一段采样？ |
| reset后立刻失败 | stand/噪声 → 高度姿态 → 阈值 | 初态是否本身无效？ |
| action shape错误 | predict输入shape → action → apply_action | 是否手动多加了batch维？ |
| NaN或MuJoCo warning | action → PD → mj_step → assert_healthy | 第一个异常发生在哪个物理量？ |
| 关节不到目标 | target/q/v → torque/限幅 → actuator映射 | 是否力矩饱和、方向错误或时间不足？ |
| 抖动 | action差分、关节速度、PD跟踪和接触 | 是目标跳变还是底层控制振荡？ |
| 跌倒/原地踏步 | 失败原因、姿态、位移、reward分项 | 是接口错误、探索不足还是任务权衡？ |
| reward涨、推理退化 | 固定种子评估、训练采样/确定性差别 | 是否只看训练曲线或挑了最好checkpoint？ |

分析AI建议时要求给出：具体触发证据、要改的消费语句、兼容性影响、对照设置、成功/失败判据。声称“换算法就解决地形适应”而没说明观测与训练分布，不足以支持实施。

本轮实际代码处理策略、保留项与未实施建议见 [审查报告](../docs/documentation_review_20261105.md)。
