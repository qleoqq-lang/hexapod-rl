# 08 模型、随机动作与边界验证

阅读入口：[inspect_robot.py](../scripts/inspect_robot.py)、[random_test.py](../scripts/random_test.py)、[test_env.py](../tests/test_env.py)。本章按验证层次组织，实际命令见 [09](09_experiments.md)。

## 三类检查各自证明什么

| 检查 | 调用链 | 当前覆盖 |
|---|---|---|
| 无RL验模 | inspect → SimRobot | 六腿/18关节、命名、stand、接地、逐关节正向响应、限位/警告 |
| 随机动作 | run → check_env → reset/step | 接口、shape、reward分项、结束类型、限矩和越界处理 |
| 有目的边界测试 | unittest → 人为设状态/非法输入 | seed、历史清零、非法动作、裁剪、坐标、姿态高度、越界、超时和失败reset |

inspect先500步零动作，逐关节各30步小正向偏移、30步回站姿，共1580控制步。单关节响应检查是方向/可动性验证，不是完整执行器辨识。

随机检查与边界测试都不是策略质量评估。随机动作摔倒而环境正确终止，可以通过；NaN或超限却不终止属于实现错误。

## random_test.run分块

```text
创建单环境
 → check_env(skip_render_check=True)
 → 创建动作rng和累计容器
 → 每回合reset(seed+ep)
 → 最多max_steps次：
     uniform(-1,1,18).astype(float32)
     → env.step → assert → 统计 → 结束则break
 → 生成报告与contract → finally close
 → 主入口写JSON
```

check_env真实执行reset/step检查，不只看函数名；skip_render_check只跳过渲染。后续回合重新reset，报告total_steps不含checker的预检查。

动作rng在run开始创建一次，跨回合连续抽样；环境初态用各自seed+ep重置。某回合提前结束会改变随后消费的动作随机数位置，不能仅凭相同reset种子认定后续动作完全配对。

## 每步检查与统计

| 条件 | 目的 |
|---|---|
| reset obs shape=(63,)、dtype=float32 | 初始接口一致 |
| observation_space.contains(obs) | step观测符合声明 |
| reward及各terms为有限值 | 防止坏评分 |
| isclose(reward,sum(terms)) | 总分与已加权分项一致 |
| 标记都是bool | Gymnasium返回类型 |
| 全测试力矩峰值≤配置上限+1e-9 | 限矩正确 |
| 越界>容差时terminated且含joint_limit | 故障能被环境处理 |
| 回合循环后存在结束标记 | 达到上限时环境确实结束 |

contains在当前±Inf Box上不是全元素isfinite的替代，底层另查物理状态；不伪造观测上下界来消除checker提示。assert在python -O下会移除，验证命令不要加-O。

episode_reward每回合清零；rewards保留所有正式单步分数；rows存回合摘要；max_torque/max_violation跨回合累计。step从0开始，报告写step+1。reward_min/max是单步范围，不是回合总分范围。

peak_torque是一个控制步内所有关节和物理子步的峰值，再取max得到全测试峰值。等于7 N·m可通过检查，但不能说明没有长期饱和。

## 报告边界

passed=True表示所执行调用与断言正常完成，不穷尽全部轨迹。warnings=0、non_finite=0、gymnasium_check=True是成功返回时填写的字段，不是全局警告计数器。

SimRobot对MuJoCo警告或物理非有限值抛异常；Gymnasium的无限Box UserWarning可能继续运行，不计入该warnings字段。异常不会自动保存passed=False报告。

完整文件含contract，终端过滤掉它以减少输出。训练前只读固定logs/inspect_robot.json和logs/random_test.json；自定义--output留档不会更新训练入口默认报告。

历史例：2026-11-03三回合各600步，总1800控制步/18000物理子步，越界0、力矩峰值7、均普通超时；终端仍有两条无限Box提示。该结果验证这些轨迹的环境行为，不包含前进速度或完整姿态评估。

## 下一步验证思维

没有采到越界轨迹就没有触发相应随机检查分支，需用边界测试补充。现有8项测试也没有覆盖所有库版本、碰撞模型、硬件或长时间运行；改动后按影响增加有意义的检查。

自查：600、80、600步的三回合total_steps=1280，不含checker。高度失败正确终止不自动使passed=False。
