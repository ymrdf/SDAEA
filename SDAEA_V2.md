# SDAEA v2：实现、运行和验证

入口是 `sdaea_streaming_v2.py`。原来的 `sdaea_online_validate.py` 和训练数据保留作对照。
本版仍然只通过视觉、身体电量及死亡学习，不读取 Godot 的 reward，也没有经验回放或世界回滚。

## 为什么这次修改算法结构

旧日志有 88,542 条交互。最后策略仍近似均匀随机：动作熵约 3.2958，随机上限为 ln(27)=3.29584。
前 10 次、后 10 次完成寿命的中位数分别为 1,225 和 1,249.5 步。单看均值会被少数长寿样本误导。
详细复算见 `SDAEA_V2_DIAGNOSIS.md`。

v1 的问题不仅是学习率小。actor 接收的是 detached 世界模型表征；生存误差没有训练视觉网络去区分有益、危险的东西。
世界模型主要拟合了最常见的微量掉血，对罕见的 ±5 HP 变化预测几乎为零。
这意味着“预测越来越准”和“行为越来越好”之间没有形成可靠联系。

## v2 改了什么

1. **生存误差直接训练视觉。** actor 与 critic 各自拥有完整视觉网络，参数相互独立。
   两条学习路径均直接看到两眼 RGB、空间位置、短期感知状态、HP 和上一动作。
   平均/最大池化的 RGB 旁路保留小色块，所有颜色通道同等处理，没有红绿标签或颜色规则。
2. **采用单一未来净内驱价值。** critic 预测未来“愉悦－痛苦”，使用无 softplus 的有符号输出。
   生理愉悦/痛苦仍分别记录，但不再让两个独立预测器的共同漂移淹没动作优势。
3. **用逐条经验适用的更新幅度控制。** 根据资格迹 L1 范数及 TD 误差自动限制有效学习率。
   每次死亡先更新、后清资格迹；不载入旧参数，不重放图像。
4. **动作保持与及时中断。** 默认一次决策保持最多 4 次 `env.step`，遇死亡或较大 HP 变化提前中断。
   每次决策结束立即更新，包括没有死亡的正常交互。用 `--hold-steps 1` 可逐次 `env.step` 更新。
   变长动作的内部信号和 bootstrap 都按实际步数折扣，旧资格迹按上一次动作长度衰减。
5. **减小恒定探索压力。** 熵项随 TD 误差幅度缩放；另保留 3% 均匀探索混合。
   采样与 log-prob 使用同一混合分布，避免行为策略与更新公式不一致。
6. **拒绝无效死亡样本。** reset 若返回 HP≤0，就再次请求 reset，不把死亡归因给尚未采取的动作。
   当前 Godot 只有 HP≤0 才真正复活身体，所以正 HP 的 done 会明确报错，避免伪造死亡训练。

首版重建下一幅图像的世界模型没有继续放进 v2 的核心循环。先用这个较小模型验证视觉与生存行为能形成闭环，
以后再加动作条件预测、长记忆和种群进化层。本版感知记忆是固定大小的视觉指数平均，并非已经学会的长期记忆。

更新规则借鉴 [Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)
中的 Stream AC / ObGD，包括资格迹、稀疏初始化和 LayerNorm。这里是针对现环境的改版，不声称复现论文基准或具有通用收敛保证。

## 内驱目标的具体含义

以原始 HP 定义 `Phi(h)=log(1+h/5)`，其中 5 是能量尺度，不是假设满血是 5。
每个 Godot step 的内部信号为：

```text
r_internal = 0.01 × alive - 2 × death + gamma × Phi(next_hp) - Phi(hp)
Phi(terminal) = 0
gamma = 0.999
```

这仍是一套人为定义的身体偏好；它没有红绿知识，也不是凭空自发产生的追求。
与 v1 指数趋近零的高血量驱动相比，log 势函数在高 HP 时仍有可分辨的能量变化信号。
势差按折扣相加会望远镜消去，在固定初始状态、正确终止/继续边界下，不会凭空奖励反复受伤再回血。
基础目标是折扣后的生存与死亡代价；它不是“数学上保证永远不死”。

当前 Godot 源码的 HP 初始值为 6、最大值为 100，红绿块是 ±5 HP。
`reset()` 会重置死亡机器人的位置和 HP，但保留方块世界；这是源码核对结果，纠正了上一版启动提示的错误描述。
若以后人类否决只产生正 HP 的 done，须在 Godot 中把它落成真正身体死亡/强制复活接口，不能混同时间截断。

## 运行

先停止旧训练进程，使用项目的 Python 环境，然后在 Godot 编辑器按 Play：

```bash
conda activate envolution
python sdaea_streaming_v2.py --run-dir runs/v2_seed7 --seed 7 --max-steps 100000
```

默认 CPU、2 线程、小型网络。可用 `--device mps` / `--device cuda`；批量为 1 时 GPU 未必更快。
`max-steps` 是实际 `env.step` 调用次数。Godot 若另设 `action_repeat`，实际物理步数会与 Python hold 相乘。
Ctrl-C 会在当前交互结束后保存。所有输出写入新目录，拒绝覆盖已有运行。

```bash
# 同动作保持长度的随机对照，单独启动一次 Godot 运行
python sdaea_streaming_v2.py --run-dir runs/v2_random7 --seed 7 --no-learn --random-actions

# 冻结已学参数的独立评估；也可改用 before_first_death.pt
python sdaea_streaming_v2.py --run-dir runs/v2_eval7 --seed 17 --no-learn --warm-start runs/v2_seed7/latest.pt --max-steps 20000

# 资格迹消融；其余条件保持一致
python sdaea_streaming_v2.py --run-dir runs/v2_no_trace7 --seed 7 --trace-steps 0
```

`warm-start` 只继承参数，不恢复过去的身体、视觉记忆或资格迹；v1 权重不兼容此架构。
冻结评估会继续更新数值感知状态，但不会修改模型参数。要比较独立初始条件，请重新启动 Godot 场景。
当前编辑器连接不能通过 Python 的 `--seed` 固定 Godot 方块随机性，多次评估不能声称是相同世界种子的配对实验。

输出包括 `metrics.csv`、`summary.json`、`initial.pt`、`before_first_death.pt`、周期快照及 `latest.pt`。
正常运行中的断连等异常会另存 `interrupted.pt`，它同样只用于参数 warm-start。
主要观察：寿命中位数和删失寿命、每单位环境步死亡数、正负 HP 事件频率、动作偏好、TD 及实际更新幅度。
动作熵降低本身不证明学会生存；必须结合冻结评估的身体结果。日志的事件计数仅根据 HP 差，不使用颜色标签。

## 已完成的验证与边界

```bash
python -m unittest -v test_sdaea_streaming_v2
python test_sdaea_streaming_v2.py --probe --steps 1500 --seed 7
python test_sdaea_streaming_v2.py --probe --steps 3000 --seed 7 --probe-gamma 0.999 --probe-trace-steps 300
```

单元/整环检查包括：非死亡更新能训练视觉、势函数折扣一致、20 步延迟资格迹、变长动作衰减、死亡清记忆、
不跨死亡执行动作、改变 Godot reward 不改变参数、冻结评估不改参数、正 HP done 不制造死亡惩罚。

另外运行了一个两动作、视觉色块位置随机、仅 HP 后果驱动的合成学习任务，再交换红绿后果继续学习。
每阶段 1,500 次在线更新，用独立亮度样本冻结评估：

| seed | 初始正确率 | 学习后 | 规则反转并继续学习后 |
| --- | ---: | ---: | ---: |
| 7 | 50% | 100% | 99% |
| 17 | 50% | 100% | 100% |
| 27 | 50% | 100% | 100% |

上面的一步选择任务使用 gamma=0.9、trace_steps=0，以匹配其短时因果结构；因此不能把它当成长资格迹导航成功。
另用 gamma=0.999、trace_steps=300、seed=7，每阶段 3,000 步，得到 50%→100%，反转后 100%。
这个长资格迹检查仍只有一个种子。合成任务只有两个动作，视觉简单，HP 事件频繁，网络也较小；
**这些结果验证学习通路与重新适应能力，不代表已经验证了真实 Godot 场景中的长期生存提升。**
