# 05 PPO：从采样、GAE到网络更新

依据本地SB3 2.9.0源码；本项目仅在 [train_ppo.py](../mujoco/train_ppo.py) 创建并调用PPO，没有重写算法。这里保留能解释实现的公式，不要求证明收敛。

## 网络与采样循环

当前新建PPO("MlpPolicy",env,...)采用ActorCriticPolicy。Actor和Critic各有[64,64]隐藏层、Tanh；Actor输出18维高斯均值及可学习log_std，Critic输出一个价值。两者使用同一63维观测，Critic不额外读取隐藏仿真信息。

```text
obs (N,63)
 → Actor：动作(N,18)、对应log_prob(N,)
 → Critic：V(obs) (N,1)
 → 裁剪动作、env.step
 → reward、新obs、回合边界 → rollout_buffer
```

价值V是从当前状态继续执行策略的未来折扣累计奖励估计，不是即时reward。采样collect_rollouts使用no_grad，参数暂时固定，但动作从分布采样；no_grad与确定性取均值是两回事。

当前默认N=1、T=n_steps=256，buffer的observations=(T,N,63)、actions=(T,N,18)，rewards/values/log_probs/episode_starts/advantages/returns各为(T,N)。保存原始采样动作及其log_prob；送环境前clip，环境历史另存applied_action。

## GAE与return：采完后构造学习目标

rollout不是完整episode。它满了以后，先用最后观测估计后续价值，再按时间倒序计算优势。

对普通连续片段，可以这样读递推：

```text
delta_t = r_t + gamma × m_t × V_old(s_(t+1)) - V_old(s_t)
A_t     = delta_t + gamma × lambda × m_t × A_(t+1)
return_t = A_t + V_old(s_t)
```

m表示是否允许跨到下一状态连接；真正终止不能把新回合价值接到旧回合。gamma=0.99折扣未来；lambda=0.95组合多步误差，调节估计偏差/方差。短rollout结束但任务未结束时可用Critic补后续；return是训练目标，不一定等于实际完整回合总分。

教学单步例：r=0.1、V旧=2、下一V旧=2.1，无终止，则delta=0.1+0.99×2.1-2=0.179。若后续A=0，该步A=0.179、return=2.179。结果较预测好，Actor获得相对有利的学习信号；reward为正不保证A为正。

### 当前SB3怎样处理超时

DummyVecEnv令TimeLimit.truncated=truncated and not terminated，保存terminal_observation后自动reset。普通超时在入buffer前执行 r += gamma×V(terminal_observation)，然后按回合边界切开GAE，避免把reset状态接进去。不要在上面的示意公式之外再次重复补值。

失败与超时同时发生时按失败处理，不走普通超时补偿。该补值不修改rewards.py，也不计入Monitor记录的原始episode回报。episode_starts用于buffer内部边界，最后一行还结合dones判断。

## 为什么能打乱、重复训练？

先用有序轨迹计算A/return，之后再将时间与环境维展平成T×N条样本并打乱。当前MLP按单个观测计算动作，监督目标已经附在样本上，mini-batch无需按原时间顺序输入。循环策略不能直接照搬这种处理，需保留序列和记忆边界。

多个epoch使同一批采样数据被充分用于优化，减少每次更新都重新仿真的成本；但策略逐步偏离采样策略，所以不能无限重复。PPO用旧log_prob和裁剪机制限制部分过大更新的优化收益，不是严格安全保证。

## 损失怎样对应行为学习

更新时evaluate_actions重新计算“当前网络对buffer中旧动作的评价”，没有在MuJoCo重新执行它们。令q=exp(log_prob_new-log_prob_old)，epsilon=0.2：

```text
policy_loss = -mean(min(q×A, clip(q,1-eps,1+eps)×A))
value_loss  = mean((return - V_new)²)
entropy_loss = -mean(动作分布entropy)
loss = policy_loss + 0.5×value_loss + 0.0×entropy_loss
```

这是当前默认无value clipping的情况。A在每个mini-batch内默认标准化；不表示obs被归一化。正负A的直觉应理解为标准化前，实际梯度还受批内相对值影响。

| 项目 | 工程意义 |
|---|---|
| policy_loss | 让相对有利动作更受偏好，同时抑制部分过大的分布改变 |
| value_loss | 让Critic更接近采样后构造的return目标 |
| entropy | 动作分布随机程度；当前ent_coef=0无额外熵鼓励，但仍随机采样 |
| clip_range | 裁剪优化目标的概率密度比，不是关节动作幅度限制 |
| approximate KL / clip_fraction | 观察新旧策略变化及裁剪活跃度，不能单列判定好步态 |

每batch：zero_grad → backward → clip_grad_norm(max=0.5) → optimizer.step。Adam更新网络参数，再用新网络采下一段。target_kl当前为None，没有KL提前停止。clip不保证所有概率、参数变化都在20%以内。

## 四种次数与默认参数

| 参数 | 当前值 | 影响 |
|---|---:|---|
| n_steps | 256 | 每环境更新前采多少步；变大增加一批数据量、内存及更新间隔 |
| workers | 1 | 总量T×N；当前DummyVecEnv顺序推进 |
| batch_size | 64 | 每次梯度计算用几条；不改变已采样总量 |
| n_epochs | 4 | 同一段学习几遍；太多可能过度拟合旧数据 |
| learning_rate | 0.0003 | 每次优化尺度；过大可能不稳、过小可能进展慢 |
| gamma / gae_lambda | .99 / .95 | 未来折扣、优势估计时间组合 |
| clip_range | .2 | PPO目标裁剪范围；不是物理保护 |

一段256条、batch64、4epoch：16次optimizer.step；本地train/n_updates每epoch加1，所以增加4。workers=2则512条、32次optimizer.step。仅epochs改8不增加交互量。

完整运行--steps 1000、N=1时采1024步、4段更新、合计64次optimizer.step。若以后启用target_kl或中断，不能机械照搬计数。

600步episode可跨多个256步rollout；更新完成不必reset。reward/loss有限且loss下降不意味着运动提升，需要独立行为评估。

## 第三方源码定位

路径均在.venv/Lib/site-packages/stable_baselines3/：

- common/on_policy_algorithm.py：learn、collect_rollouts、超时补值。
- common/buffers.py：compute_returns_and_advantage、get、展平和打乱。
- ppo/ppo.py：train的损失、epoch/batch与优化。
- common/policies.py：ActorCriticPolicy、默认网络、predict。
- common/vec_env/dummy_vec_env.py：顺序step、终止观测与自动reset。

学习依据：[PPO原论文](https://arxiv.org/abs/1707.06347)、[GAE原论文](https://arxiv.org/abs/1506.02438)。项目行为以当前已核对库源码为准。保存恢复见 [01](01_training_flow.md)，实际读日志见 [09](09_experiments.md)。
