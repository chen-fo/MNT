# MNT 本地观战

可视化位于 MNT 仓库的 `feat/local-visualization` 分支。使用 Pygame 绘制二维棋盘，
不修改环境移动、射击、奖励或观测规则，不启动训练。可以只保留在本机；服务器不需要安装 Pygame。
已经运行的服务器实验继续使用原来的代码版本，避免影响续训时的源码校验。

## 1. 安装与随机对局

在本机 PowerShell 中运行：

```powershell
conda activate mtapro-mnt
cd E:\cfk\proj\MNT
python -m pip install -e ".[visualization]" --no-build-isolation
python examples/watch.py --config configs/visualization/mnt_2v2.json --paused
```

安装项仅增加可选 Pygame 依赖，不包含 Torch 安装或升级。
安装后也可以从任意目录运行 `mnt-watch`，或 `python -m mnt_env.visualization.cli`。
没有传入模型的一方使用随机合法动作；合法动作仍可能导致碰撞、出界等失败。

## 2. 查看已经训练的团队模型

先将服务器的 `.pt` 文件下载到本机，例如 `MNT/outputs/models/`。
这里需要**冻结导出的团队策略**，如 `subject_ippo_gru64_seed0_round2.pt` 与
`best_round2.pt`，或者带训练阶段标识的对手模型；`resume.pt` 是续训包，不能用于观战。
优先选择同轮次配套的 subject/opponent；混搭也能观战，但不是原训练组合。

```powershell
python examples/watch.py --subject outputs/models/subject_ippo_gru64_seed0_round2.pt --opponent outputs/models/best_round2.pt --paused
```

模型入口通过可选的 MTAPRO 加载接口调用已有网络，不复制算法实现。
当前 conda 环境需已安装兼容 Torch 和 MTAPRO；如果尚未安装 MTAPRO 且依赖已齐全：

```powershell
python -m pip install -e E:\cfk\proj\MTAPRO --no-deps --no-build-isolation
```

环境参数默认从 checkpoint 恢复，包括双方数量、射程和奖励参数。
同时指定 `--config` 或两个 checkpoint 时会检查环境参数是否一致，
也会检查双方各自的观测范围、动作数与团队大小。此分支支持原版 blind sonar，不能观看 typed sonar 模型。
这些检查不等于源码版本相同；复现实验时仍应使用与训练一致的 MNT 规则版本。

默认 CPU 推理、按策略概率采样动作，足够驱动本地窗口；可通过 `--device cuda:0`
选择 GPU，或通过 `--deterministic` 改为最大概率动作。每个团队成员的 GRU 记忆独立维护，每局清空。
网络只接收各自的环境观测；观战地图的全局信息不会输入策略。

## 3. 看什么、怎么操作

- 蓝色 A 是我方，红色 E 是敌方；短白线表示当前朝向，绿色 G 是终点，金色叉号是地雷。
- 黄色线是刚执行的射击，停在真实阻挡单位或射程末端；暂停时保留，下一步刷新。
- 同格单位分散绘制在同一个格内；灰色划线单位是已退出单位的标记，不是新增障碍。
- 右侧显示每个单位的状态、当前步奖励、整局累计奖励，以及最近事件。
- 选中单位显示实际观测：朝向、终点相对坐标、五条声纳的最近物体距离。
  `--` 表示该方向范围内无探测，声纳不识别物体类型。全局地图属于人类观战视角。
- 出界、踩雷、友方碰撞、敌方碰撞、炮击、超时等使用环境返回的事件名称展示。
- 这里的奖励是**环境原始个人奖励**，不包含 MTAPRO 训练侧私有奖励，不能直接与训练总回报曲线比较。

快捷键：`Space` 暂停/继续；`→` 暂停并执行一步；`+/-` 调速；`Tab` 切换单位；
`N` 开始下一局（未结束的当前局将丢弃）；`Esc` 或关闭窗口退出。
结束后画面保留，按 `N` 才开始下一局。可用 `--speed 10 --seed 42` 指定速度和起始种子。

## 4. 无窗口验证与结果保存

```powershell
python examples/watch.py --headless --episodes 3 --seed 42 --output outputs/visualization/random_matches.json
```

也可附加双方模型参数，执行短对局推理。随机对局不需要 Torch/MTAPRO；无窗口模式不需要 Pygame。
`--episodes` 只控制无窗口模式；窗口模式手动切换对局。
`--output` 在每局完整结束后写入种子、步数、个人回报、最终状态等，关闭或跳过的半局不计入。

## 5. 文件职责

```text
src/mnt_env/visualization/
    cli.py          # 参数解析、窗口控制循环
    renderer.py     # 仅绘制，不推进环境
    geometry.py     # 根据调试快照还原射线
    session.py      # 对局推进与个人回报记录
    policies.py     # 随机控制器、可选冻结团队模型桥接
examples/watch.py   # 薄启动入口
configs/visualization/mnt_2v2.json
tests/test_visualization.py
docs/visualization.md
```

渲染使用现有 `get_debug_info()` 返回的独立快照，原 `env.render()` 文本接口保持可用。
Pygame API 参考：[绘图](https://www.pygame.org/docs/ref/draw.html)、[窗口事件](https://www.pygame.org/docs/ref/event.html)。
