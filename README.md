# MNT Benchmark

新增本地 Pygame 观战工具：支持随机对局、加载冻结团队策略、暂停与逐步查看。
安装和操作见 [本地可视化说明](docs/visualization.md)，模块变化见 [项目成长记录](WORK_LOG.md)。

二维离散雷区撤离／坦克拦截环境。双方所有 agent 都接受外部策略的动作，
采用 **PettingZoo Parallel API** 同时推进，每个 agent 独立获取局部观测与个人奖励。
本项目只提供环境、随机策略示例与规则测试，不包含训练算法。

## 安装与运行

```powershell
conda env create -f environment.yml
conda activate mtapro-mnt
python examples/random_match.py --config configs/default.json --seed 42 --render
python -m pytest -q
```

已有兼容 Python 环境可运行 `python -m pip install -e ".[dev]"`。
`environment.yml` 用于创建独立环境，与 SMACLite 的 `mtapro-smac` 环境隔离。
本次 Windows / Python 3.10.21 验证使用的精确 Python 依赖保存在
`requirements/py310-win64.txt`；复现时可先 `python -m pip install -r requirements/py310-win64.txt`，
再 `python -m pip install -e . --no-deps`。
随机对战可使用 `--episodes 10 --output outputs/demo.json` 保存个人回报与结束状态。

```python
from mnt_env import parallel_env

env = parallel_env(num_allies=2, num_enemies=2, shoot_range=3)
observations, infos = env.reset(seed=42)
while env.agents:
    # 示例随机策略；算法应当给每个 agent 传入自己的 observations[agent]。
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

`possible_agents` 固定为 ally_0... 后接 enemy_0...；`agents` 只包含当前还能行动的单位。
step 的五个返回值都是按 agent ID 索引的字典，包含本步刚退出者的终局结果；下一步不再返回它们。
双方都必须显式提交动作，没有规则控制的隐藏敌方，也不自动共享策略参数、观测或奖励。

## 运动、射击与同时结算

默认 16×16、2 个我方、2 个敌方、8 个地雷，声纳距离 8 格、敌方射程 3 格，任务期限 100 步。数量及射程等可配置。
坐标为 `(x, y)`，y 向下；朝向 0...7 顺时针依次为北、东北、东、东南、南、西南、西、西北。

- 我方 `Discrete(5)`：0 左转90°并移动，1 左转45°并移动，2 直行，3 右转45°并移动，4 右转90°并移动。
- 敌方 `Discrete(8)`：0...4 同上；5 左偏45°射击，6 正前方射击，7 右偏45°射击。
- 射击不改变本步位置和车身朝向；没有额外的停留动作、弹药限制或冷却。
- 这是网格运动：对角也走一个格步；不是恒定欧氏速度或真实无人机动力学。
- 我方越界失效；敌方越界移动被阻挡，位置不变，但保留该动作产生的新朝向。

每一步按以下固定顺序结算：

1. 校验全部动作后，同时执行移动，射手保持原位。
2. 我方越界失效；按移动后的终点格判定碰撞。交换位置、途中路径交叉不算碰撞。
3. 同格有 n 个我方和 m 个敌方时，我方全部失效，敌方存活，每个敌方获得 n*r/m。
   只有我方且人数至少为 2 时，它们全部失效。敌方可重叠。
4. 尚存我方踩雷失效，敌方免疫。碰撞优先于踩雷；一个我方只扣一次失败奖励。
5. 尚存我方到达终点立即获得个人奖励并撤离，不再挡炮或遭受本步炮击。
6. 所有炮弹根据同一份状态计算命中，再统一更新死亡。
   炮弹可穿雷，遇第一个在场坦克或地图边界停止；敌方挡炮但不受伤，我方一击失效。
   同一我方被 k 个射手命中，每个射手获得 r/k。无关敌方获得 0。
7. 第 100 步照常完成上面所有事件；之后仍在场的我方超时失败，扣 -r，不给敌方击杀奖励。

射程按格步计算，最大距离包含端点；炮弹从相邻格开始检测，同格敌方不挡彼此出膛。
本步被击中的单位仍遮挡同一时刻的其他炮弹，下一步移除；没有残骸阻挡。
地雷不会消失，终点不是声纳或炮弹的障碍物。

任一方没有仍在场的单位，或到达任务期限，整局终止。
当前敌方免疫且没有被击毁途径，所以通常因我方全部死亡／撤离或期限到达而结束。
每个我方独立到达、独立得分，不设“任意一人成功即全队成功”。

## 个人奖励与状态

- `failure_penalty=r>0`，默认 10。我方越界、碰撞、踩雷、被击中和超时均获得 -r。
- 默认到达奖励保留原型公式 `target_reward * gamma**steps`，默认 `50 * 0.95**steps`。
  这是环境内的时间偏好，不是算法折扣；若想要固定到达奖励，配置 `gamma=1.0`。
- 敌方只因自己参与的碰撞／炮击击杀获得奖励，不存在团队复制奖励或团队总回报。
- 普通步奖励为 0；退出单位不再行动、领步奖励或被重复扣分。
- `active / evacuated / dead / timed_out` 分别记录，撤离不等同于死亡。

100 步是任务本身的期限，故到期 `terminations=True`，`truncations=False`。
全局结束时仍存活的敌方也收到 termination，表示回合结束而非坦克被击毁。
结束后必须 reset，禁止继续 step。不同大小地图不保证随机布局一定可解；评估应固定场景和种子。

## 观测与全局状态

每个 agent 的 `Box(float32, shape=(9,))` 为：

```text
[heading, target_dx, target_dy,
 sonar_left90, sonar_left45, sonar_front, sonar_right45, sonar_right90,
 elapsed_steps]
```

默认每束声纳只沿对应射线查找 1...8 格；不是半径 8 内的全景视野。
最近物体返回 `1 / 格步距离`，范围内没有物体返回 0。
墙、地雷、我方、敌方的编码完全相同；不返回类别、数量或 ID，后方物体被遮挡。
敌方站在雷上也只产生最近占用格的一个信号。第 9 格及更远不影响声纳。
射线从相邻格开始，因此不会额外报告与自己同格的敌方同伴。
保留精确目标相对坐标、自身朝向和步数；不提供会泄漏障碍类别的动作掩码。
终局退出单位返回零观测（步数保留）。

`infos[agent]` 仅含自身状态、事件名称、奖励分项、步数与回合结束标志。
它不含其他 agent 的位置、观测或身份，也不含全图。自己的击杀奖励可能反映共同贡献者数量，
这是任务奖励本身的信息，不是额外共享信息。

`env.state()` 为可选全局接口，返回拷贝；局部观测实验不应把它传给 actor。
布局为：`mines_map[x,y]` 按 C 顺序展平、`target_x,target_y,steps`、
按 possible_agents 顺序逐个附加 `[team,x,y,heading,status]`。
team 为我方0／敌方1，status 为 active0／evacuated1／dead2／timed_out3；有对应 `state_space`。
`get_debug_info()` 另提供完整位置、事件及命中归因，仅供调试、回放或评估记录。

## 固定布局与复现

`reset(seed=..., options=...)` 支持 positions、headings、target、mines。
positions 和 headings 按 possible_agents 排列，必须为整数；mines=[] 表示无雷。
其他 reset options 字段作为外层框架元数据忽略；已知布局字段的值仍严格校验。
未指定的字段使用环境自己的 RNG 生成，reset 不修改 Python 或 NumPy 全局随机状态。
默认随机起点互不重叠，先确定终点再采样其他对象，不用无限拒绝采样。
目标缺省从离边界两格的内部区域选择，没有可用位置时立即报错。
自定义起点允许敌方互相重叠或站在雷上；不允许我方起始重叠／踩雷、任何单位起始占据终点。
自定义地雷坐标覆盖 num_mines 的数量设置。

## 文件组织与旧接口

```text
src/mnt_env/             环境、配置、布局生成、随机对战入口
configs/default.json    标准默认场景
examples/random_match.py
tests/                  规则和官方 Parallel API 测试
legacy/Mnt_navigation.py 原文件备份，不作为新环境运行
Mnt.py                  新环境导入入口
outputs/                本地输出，Git 忽略
```

安装后 `from Mnt import Mnt` 仍可在项目目录使用，但 Mnt 现在就是 MntEnv，使用字典接口。
旧的联合动作数组／向量奖励 Gym 接口没有冒充兼容：请改用 `num_allies/num_enemies`、
`sonar_range`，并通过 `env.agents` 收集所有在场单位动作。
旧的 contact_radius、terminate_on_target、step_penalty 已移除，使用上述确定的任务规则。

接口参考：[PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)，
验证参考：[PettingZoo 环境测试](https://pettingzoo.farama.org/content/environment_tests/)。
