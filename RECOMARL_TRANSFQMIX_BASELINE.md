# ReCoMARL 论文的 TransfQMix 基线记录

更新日期：2026-10-03。本文记录**实验设计与代码状态**，不是实验结果。当前论文 `D:\Desktop\6a8b147b3fef5bb91032d90d\Main.tex` 的主表尚未列出 TransfQMix；如果最终纳入比较，应在论文正文、图表、补充材料和统计脚本中同步增加，不能直接把未运行的基线描述为已比较。

## 要回答的问题

论文研究的是一个已训练团队在成员持续缺席后的协调恢复。完整团队先训练至 `T_F=5,000,000` 环境步；此后每个新回合先生成完整团队，物理移除 **allied slot 3** 对应的实际 SC2 单位，之后才开始策略动作。其余单位沿用前 5M 步学到的参数，继续训练额外 `B=5,000,000` 环境步。固定的是槽位编号，随机队伍和位置可能使该槽位对应不同单位类型；这里不再使用 QMIX 关键智能体选择器。

TransfQMix 作为基线检验：原论文的 Transformer agent + 单调 Transformer mixer 在**相同物理移除、交互预算与评估协议**下能恢复多少；不能把它描述成 ReCoMARL 的可变人口专用结构。当前代码保留原 agent、mixer、TD(λ) 学习器、Adam 优化器和目标网络更新逻辑，扩展的仅为 SMACv2 数据适配、可执行动作的目标对齐、缺席 slot 的固定张量占位，以及外层恢复协议。跨方法比较时，各方法也必须移除 slot 3；旧的“冻结选择器找关键智能体”协议所得曲线应单独标注，不能直接混作同一组结果。

## 固定条件与参数来源

| 项目 | 当前设定 | 依据／说明 |
| --- | --- | --- |
| 地图 | Terran、Protoss、Zerg × 10v10、15v15、20v20 | 论文九种 SMACv2 任务；使用对应 `10gen_*` 图并改 `n_units`/`n_enemies`。 |
| 队伍抽样 | 各族三种兵权重 `0.45/0.45/0.10`、随机位置 | 与现有 ReCoMARL 任务定义对齐。 |
| 对手难度 | SMACv2 字符串难度 `"7"` | 环境配置沿用 VeryHard；没有私自降低对手难度。 |
| 采样与更新 | 8 个并行客户端；每收集 8 个完整回合做 1 次 learner 更新 | 论文的 off-policy 计数约定；故障回合整批丢弃，不计胜负、不入回放池、不增加有效 `t_env`。 |
| 完整队伍／恢复预算 | 5M＋5M 环境步 | 实际切换点是首次达到 5M 后的下一回合边界，审计文件记录实际步数；末尾可能因整批回合越过名义 10M。 |
| 网络 | `emb=32, heads=4, depth=2, mixer_emb=32, mixer_heads=4, mixer_depth=2, qmix_pos_func=abs` | 原仓库 `transf_qmix_smac.yaml`。未替换为 ReCoMARL 网络。 |
| 优化 | Adam `lr=0.001`、`td_lambda=0.6`、`batch_size=32`、回放容量 5000、每 200 回合更新目标网络 | 原 TransfQMix 配置；ReCoMARL 的 `batch_size=128` 不应直接套到该基线。 |
| 完整队伍探索 | ε 从 1.0 线性降至 0.05，100k 步 | 原 TransfQMix 配置。 |
| 恢复探索 | ε 在切换点设为 0.24，20k 步线性降至 0.05 | **新增的匹配恢复协议**，不是原 TransfQMix 论文参数；与论文补充材料的 matched-exploration 设定一致。若放入主比较，必须明确披露；如需方法原生 `ε=0.05`，另跑对照，不要混合曲线。 |
| 评估 | 每 10k 有效环境步做 32 个 greedy 回合；同一份固定配置清单 | 论文评估协议。首次 post-failure 评估在实际移除后首个训练批次之后，并强制记录预算末尾评估。 |
| 种子 | 计划每任务 5 个训练种子 | 代码支持单种子运行；尚未完成五种子实验。 |

### 固定移除槽位与评估清单

恢复实验直接移除 allied slot 3，不需要额外训练 QMIX selector 或传入 `mixer.th`。`results/recovery_audit/*.json` 记录 `failure_agent_id=3`，每次移除后评估也记录 `removed_slots`。跨方法应核对它们使用同一槽位、相同失效边界和相同训练/评估预算。

恢复运行首次评估会导出 `results/eval_manifests/*.json`，内含 32 组实际 team/position 配置及校验元数据；同一任务/种子的后续运行传 `eval_manifest_path=...`。当前 TransfQMix 会锁定该清单用于其所有后续评估。其他基线必须增加读取这份清单的能力，并核对每个回合均移除 slot 3；仅凭同一个随机种子，不能保证不同实现实际生成相同配置。未完成这些跨仓库核对之前，不应称实验“已严格公平配对”。

## 数据流与实现位置

1. `src/config/algs/transf_qmix_recovery_smacv2.yaml`：`failure_agent_id: 3` 定义每回合要移除的 allied slot；旧的 selector 训练配置留作独立研究，不用于当前恢复实验。
2. `src/envs/smac_v2/StarCraft2Env2Wrapper.py`：原生扁平观测转为 entity token；SMACv2 医疗兵治疗动作的目标是**友军**，适配器为其将友军 token 放在动作头读取的位置；普通单位保持敌军 token 在前。全局 state 调整到原 SMAC entity-state 的 health/position/cooldown 顺序。环境 reset 后调用 SC2 `_kill_units` 移除 slot 3 的单位，验证指定 tag 消失且没有误杀其他友军。移除发生在首个 policy step 之前；人工死亡不计入战斗伤亡。
3. `src/runners/parallel_runner.py`：8 worker 并行采样。恢复阶段给固定容量 batch 写入 `participating_mask` 和 `removed_agent_id`，并检查 worker 确实返回 slot 3；缺席 slot 仅有 no-op 占位。单个 worker 的环境异常会丢弃**整个未完成的 8 回合批次**并最多重试 3 次；超过上限明确报错。失败批次不会写入经验池或有效步数，但不是无限容错保证。
4. `src/learners/nq_transf_learner.py`：只把初始被移除 slot 的 executed utility、target utility 与喂给 mixer 的 hidden 向量清零；其余正常战斗阵亡不按“初始移除”处理。Transformer mixer 结构及注意力本身**没有新增加存活掩码**，保留固定 slot 是该基线的架构局限，不应把它写成 ReCoMARL 式的动态参与者聚合。
5. `src/run/run.py` 与 `src/utils/recovery_metrics.py`：保留网络、优化器、目标网络及 replay 跨 5M 边界连续训练。记录 `test_battle_won_mean` 和 post 阶段的 `test_post_failure_battle_won_mean`；从未平滑的真实评估点计算 `J_pre`（名义 `[4.7M,5.0M)`）、`J0`（首个 `b>0` 的移除后评估）、`AUC_rec` 与 `AUC_gain=AUC_rec-J0`。只在末尾存在跨越 `B` 的评估点时线性插值，不外推。

## Linux 服务器运行方式

以下以 Terran 20v20、种子 1 为例；先在服务器的该仓库目录和可导入 PySC2/SC2 的环境中执行。

```bash
export SC2PATH="$HOME/StarCraftII"
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=transf_qmix_recovery_smacv2 --env-config=sc2_v2_terran with seed=1 env_args.capability_config.n_units=20 env_args.capability_config.n_enemies=20 use_cuda=True use_tensorboard=False
```

正式运行前可用固定 slot 3 做跨越失效边界的小规模冒烟（这只是协议/接口检查，不能用于论文曲线）：在上述恢复命令的 `with` 参数末尾追加 `failure_t_env=200 recovery_budget=200 t_max=400 test_interval=100`。通过后重新从零启动正式 5M＋5M 运行。

其它任务换 `sc2_v2_protoss` / `sc2_v2_zerg` 与 `n_units=n_enemies=10/15/20`。要与已有基线配对，再传 `eval_manifest_path=/绝对路径/到/同任务同种子的32配置清单.json`；不同队伍规模、种子不要混用清单。默认 `failure_agent_id=3`，也可在命令行显式传入相同值，便于审计。

## 结果文件和验收条件

- Sacred 指标：`results/sacred/<map>/<algorithm>/...`；模型：`results/models/<run>/<step>/`；恢复审计：`results/recovery_audit/<run>.json`；固定评估配置：`results/eval_manifests/<run>.json`。
- 提交结果前核对每任务/种子：8 worker、5M 前没有移除、之后每回合 slot 3 单位物理消失、配置清单 SHA 跨方法一致、post 32 回合目标 slot 配对、丢弃批次数量、有效交互步数与 AUC 端点。
- 统计按**每个训练种子先算指标，再对 5 个种子求平均和样本标准差**；不要把 32 个 eval 回合当 32 个训练种子。
- 历史开发阶段曾完成本机 Python 全源编译、纯 Python AUC 边界/插值测试，九种 race×人数的环境元数据/agent/mixer 张量维度前向检查，以及合成 batch 的缺席掩码 learner 更新检查。这些检查**不能证明当前固定 slot 3 版本已通过服务器上的真实环境 reset/step 与 8 客户端并发**，也没有 10M 训练结果。服务器更新代码后先做上面的短程跨失效边界冒烟，再正式跑九任务。
