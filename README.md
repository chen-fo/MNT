# UAV Pursuit

二维网格 UAV 逃逸／追捕环境，使用 **PettingZoo Parallel API** 同时推进双方动作。
逃逸者独立前往终点；追捕者通过同格或上下左右相邻捕获逃逸者。
这是个人收益博弈，支持 1v1、2v2 和非对称人数；项目提供环境与示例，不包含训练算法。

## 安装与运行

```sh
python -m pip install -e ".[dev]"
python examples/random_match.py --config configs/default.json --seed 42 --render
python -m pytest -q
```

也可使用独立 Conda 环境：

```sh
conda env create -f environment.yml
conda activate uav-pursuit
uav-demo --episodes 10 --output outputs/uav_demo.json
```

源码未安装时，可在仓库根目录运行 `PYTHONPATH=src python -m uav_env.demo`。
依赖范围见 `pyproject.toml`；没有宣称跨平台锁定复现，旧分支的平台验证记录不适用于此版本。

```python
from uav_env import parallel_env

env = parallel_env(num_evaders=2, num_pursuers=2, vision_radius=4)
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

1v1 使用 `num_evaders=1, num_pursuers=1`。
`possible_agents` 固定按 `evader_0...`、`pursuer_0...` 排列。
`agents` 只包含当前仍能行动的单位；撤离或被捕者当步收到最后一次结果，之后从 `agents` 移除。
双方全部动作都由调用方显式提供，没有环境内隐藏的对手策略。

## 地图、动作与结算

默认 16×16 空网格、2 个逃逸者、2 个追捕者、视野半径 4、任务期限 100 步。
坐标为 `(x, y)`，x 向右、y 向下。双方均为 `Discrete(5)`：

| 动作 | 效果 |
| --- | --- |
| 0 | 不动 |
| 1 | 上，`(0, -1)` |
| 2 | 右，`(1, 0)` |
| 3 | 下，`(0, 1)` |
| 4 | 左，`(-1, 0)` |

每步移动一个格子，无朝向、斜向移动、地雷、内部障碍或射击。
越界移动被阻挡，单位留在原格，不扣分。同队单位可以重叠，不发生友军碰撞死亡。

结算顺序固定为：

1. 校验所有动作，然后同时移动所有在场单位。
2. 逃逸者若到达终点，立即获得个人到达奖励并撤离。
3. 对剩余逃逸者，按移动后的坐标判定捕获：与任意追捕者的**曼哈顿距离 ≤1**，即同格或上下左右相邻。斜对角不算。
4. 多个追捕者同时捕获同一个逃逸者时，该逃逸者只失败一次，捕获者平分该次捕获奖励。捕获归因使用同一份移动后状态，不受遍历顺序影响。
5. 第 `max_steps` 步照常执行到达与捕获；然后仍在场的逃逸者超时失败。

**到达优先于捕获**：即使追捕者同时进入终点或位于终点邻格，本步到达的逃逸者仍成功撤离。
这只决定单步结算的优先级，没有终点周围额外的免捕获区域。
若追捕者提前占据终点并持续停留，逃逸者仍会在进入终点前的相邻格被捕，因此仅到达优先不能消除长期守点封锁。
该规则不保证逃逸方对任意追捕策略都能取胜。
多个同队逃逸者可以同一步到达并分别成功。

捕获仅根据移动后的位置判定，不进行连续路径碰撞检测。
正常在场双方不会在一步开始时相邻，因此四方向一步移动下无需额外的相邻位置交换规则。
追捕者不会被击毁。所有逃逸者已撤离／被捕，或达到期限时，整局结束。

## 个人奖励与终止

| 事件 | 逃逸者 | 追捕者 |
| --- | --- | --- |
| 到达 | 本人 `+target_reward`，默认 +10 | 0 |
| 被捕 | 本人 `-failure_penalty`，默认 −10 | `capture_reward`，默认 +10，由实际捕获者平分 |
| 超时 | 本人 `-failure_penalty` | 0 |
| 普通步、越界阻挡 | 0 | 0 |

一次捕获没有贡献的追捕者得 0；队友撤离也不复制奖励。
同一追捕者捕获多个逃逸者时，累加自己的各次贡献奖励。
没有环境内的成功奖励时间衰减；训练算法可自行配置折扣。
当前奖励表达个人撤离／捕获贡献，不是同队共享奖励，也不是两队零和效用。
若评价总体拦截率，需注意“让对手超时”和“让对手撤离”对追捕者都没有奖励。

状态分别为 `active / evacuated / captured / timed_out`。
退出的逃逸者不再行动、得分或重复受罚；其他单位继续行动直到整局结束。
全局结束时，仍存活的追捕者也收到 termination，表示回合结束，不表示它被捕。
任务期限属于任务失败条件，因此到期 `terminations=True, truncations=False`。
结束后继续 `step()` 会报错，需要 `reset()`。

`end_reason=all_evaders_resolved` 表示逃逸者全部已撤离或被捕，可能是混合结果；
`deadline` 表示至少有一个逃逸者实际超时。成功率应根据各单位的最终状态统计，不能只看停止原因。

五个 step 返回字典都按步前 `agents` 索引，包含本步退出者的终局结果。
下一步应按更新后的 `env.agents` 收集动作，不能继续给已退出者提交动作。

## 局部视野与全局状态

每个单位的观测是 `Dict`，包含两个 `float32` 数组：

| 字段 | 形状 | 内容 |
| --- | --- | --- |
| `local_map` | `(4, 2*r+1, 2*r+1)` | 以自身为中心、半径 `r=vision_radius` 的方形窗口 |
| `self` | `(6,)` | `[x, y, target_dx, target_dy, elapsed_steps, team]` |

`team` 为逃逸方 0、追捕方 1。`self` 使用原始网格坐标和步数，未归一化。
任务终点位置始终已知，因此目标相对坐标不受视野限制。
其他单位的坐标只在局部窗口内出现，不自动共享队友视野。

`local_map[channel, row_y, col_x]` 的中心是 `[r, r]`，四个通道为：

1. 地图边界：窗口中越界的格子为 1。
2. 其他同队单位数量，不包括观测者自身。
3. 对手单位数量。
4. 终点：终点在窗口内时对应格子为 1。

方向不受朝向影响；对角格也在方形视野内，但捕获仍只使用四邻格。
没有内部障碍或遮挡；一个格子可以同时在不同通道有值。
允许同队重叠，因此保留单位数量而非单一占用标记。
离场单位不再出现在局部地图。终局退出逃逸者返回零地图和零自身特征，但保留步数与阵营。

`infos[agent]` 仅含自身 `status / events / reward_components / steps / episode_done`。
它不返回其他单位的隐藏位置、身份或观测。
共享捕获的个人奖励本身可能反映贡献者数量，这属于奖励信息。

`env.state()` 提供可选全局状态和对应 `state_space`，布局为：

```text
[target_x, target_y, steps,
 team_0, x_0, y_0, status_0,
 team_1, x_1, y_1, status_1, ...]
```

单位顺序为 `possible_agents`，状态编码为 active=0、evacuated=1、captured=2、timed_out=3。
返回值是拷贝。`get_debug_info()` 另提供完整位置和事件归因，仅用于调试、回放或评估。
集中训练时可将 state 给 critic；局部观测 actor 应只使用自己的观测。

双方动作空间相同，同队可以共享策略参数；跨阵营共享策略时应使用观测中的角色信息。
固定 agent 轴的训练器需要按 `possible_agents` 补齐离场者并记录活动／终止掩码。
改变人数会改变全局状态维度，改变视野半径会改变局部地图形状，不能无适配混入固定输入维度网络。
普通单智能体 Gym 训练器需要额外封装对手策略；本项目没有声称已接通任何具体算法训练链路。

## 布局与复现

`reset(seed=..., options=...)` 支持 `positions` 和 `target`：

```python
observations, infos = env.reset(seed=7, options={
    "positions": [[1, 2], [1, 4], [5, 2], [5, 4]],
    "target": [3, 3],
})
```

positions 按 `possible_agents` 排列。已知字段严格要求整数、合法形状和地图范围；其他框架元数据字段忽略。
初始单位不能重叠或占据终点，敌我初始曼哈顿距离必须大于 1。
每个逃逸者到终点的曼哈顿距离不得超过任务期限。
未指定字段由环境自身 RNG 生成，不修改 Python 或 NumPy 全局随机状态。
随机布局最多尝试 1,000 次，无法满足约束时明确报错，不会无限拒绝采样。
这些约束保证初始状态有效且存在不考虑对手的静态路线，不是对抗必胜保证。

可复现的固定 1v1 演示：

```sh
uav-demo --scenario evacuation --render
uav-demo --scenario capture --render
uav-demo --scenario timeout --render
```

三个场景分别演示同一步双方进入终点但逃逸者成功、同格捕获、期限到达仍未撤离。
`--scenario` 自带固定配置和动作，与 `--config` 互斥。
它们用于核对规则，不是内置对手策略或性能基线。

## 文件组织

```text
src/uav_env/             环境、配置、布局和演示入口
configs/default.json    默认 2v2 场景
examples/random_match.py
UAV.py                  项目目录下的导入入口
tests/                 规则、布局和官方 API 检查
outputs/                本地输出，Git 忽略
```

该分支使用新的 UAV 配置和接口名称，旧版本保留在 master。
接口规范参考 [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)。
