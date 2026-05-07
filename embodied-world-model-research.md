# 面向具身任务的 World Model 研究调研与方案整理

> 整理时间：2026-05-07  
> 主题：从 world model 角度切入具身场景训练，重点关注 2025 年至今的 world model、world action model、VLA/world model 融合、2D/3D/4D 表征、多视角数据、任务设计与实验方案。  
> 说明：用户后续特别要求扩充 Pi0.7 系列、Fast-WAM、DreamDojo，并整理这些工作的 world model backbone 与架构设计，因此本文同时覆盖 2025 年和 2026 年初的相关工作。

---

## 0. 用户原始问题

用户希望从 world model 角度切入具身相关场景训练，但还没有想清楚方向和未解决问题，因此提出以下问题：

1. 现在的 world model 的研究进展到哪一步了？一般用什么数据进行训练的？
2. world model 在训练完成之后会被拿来做什么？world model 能在其他框架流程中起到怎样的一个作用？
3. 2D、3D、4D 的 world model，它们在训练流程、训练数据、应用场景中分别有什么差异？
4. 如果想要给具身相关的 task 提供合理支持，应该如何设计 world model 输出的信息，信息又能够起到什么样的赋能作用？
5. 具体来说，如果以具身 3-view 的数据形式作为输入，我们拿到了怎样的一种终态数据形式表征，才能认为一个具身任务是 100% 解决了的？
6. 具体来说，如果一个大小脑结构的 VLA，或者别的具身结构，world model 一般在其中扮演什么角色，它输出怎样的一种数据形式来在 workflow 中起到作用？这种数据形式、数据量能否被增广或增强，比如从 single 2D image 到 2D video，2D 到 3D？增强之后带来的优势是什么？
7. 如果沿用第六个问题来计划开展研究，应该如何设计实验或对比来论证方法相较其他 world model 存在优势？如果最终落到具身实验上，应该如何做？
8. 如果并不想做普适巨大范围的具身 task，而只把问题限制到一到两个或几个具体 task，例如叠衣服、叠箱子，推荐做什么 task？哪些任务有更多数据集或相关文章？

后续补充要求：

- 扩充更多调研广度。
- 覆盖 Pi0.7 系列、Fast-WAM、DreamDojo。
- 说明这些工作采用的 world model backbone 和架构设计。

---

## 1. 总体判断

2025 年至今，world model 的主线已经从“生成好看的未来视频”转向：

1. **可交互**：给定 action 或 latent action 后，模型能预测世界如何变化。
2. **可控制**：未来状态必须严格响应动作，而不是只生成视觉上合理的视频。
3. **可几何验证**：对具身任务，2D RGB 不够，深度、3D object state、contact、stability、4D scene 变得重要。
4. **可被策略消费**：world model 输出不应只给人看，而要能服务 VLA、policy、planner、verifier、MPC、failure recovery。
5. **可闭环**：推理时是否显式生成未来已成为争议点。Fast-WAM、UVA、VPP 表明很多收益可能来自训练时 video/world modeling，而不是推理时真的生成未来。

一个更精确的趋势划分是：

| 方向 | 代表工作 | 核心思想 |
|---|---|---|
| 通用 world foundation model | Cosmos、Genie 3、Vid2World | 大规模视频预训练，获得视觉和物理先验 |
| robot world model / data engine | GigaWorld-0、DreamDojo、DreamGen | 用 world model 生成或模拟机器人数据 |
| WAM / World Action Model | DreamZero、LingBot-VA、Fast-WAM、Motus、WorldVLA | 同时建模未来视觉和动作 |
| VLA + world model context | Pi0.7、Goal-VLA、PhysicalAgent | world model 生成 subgoal image/video，VLA 执行动作 |
| 3D/4D embodied world model | TesserAct、WristWorld、4D-VLA、Embody4D、X-WAM | 从 2D 视频走向动态 3D/4D 场景和多视角一致性 |

对具身研究最有价值的切口不是“再做一个更大视频生成器”，而是：

> 面向 VLA/策略训练的多视角 action-conditioned 3D/4D world model：输出可验证、可规划、可被策略消费的未来状态。

---

## 2. 问题 1：world model 研究进展与训练数据

### 2.1 研究进展

当前 world model 已经发展到以下阶段：

#### 2.1.1 从 passive video generation 到 interactive world model

早期视频模型主要根据 text/image 生成未来视频；现在更关注：

```text
past observations + action / command / trajectory -> future world state
```

典型工作：

- Vid2World：把预训练 video diffusion model 改造成 causal、action-conditioned interactive world model。
- DreamDojo：用 latent actions 从大规模人类第一视角视频中学习可交互世界动态。
- Cosmos：提供 Physical AI 的 world foundation model 平台。

#### 2.1.2 从 VLA 到 WAM

VLA 通常学习：

```text
observation + language -> action
```

WAM 进一步学习：

```text
observation + language -> future video/state + action
```

典型工作：

- WorldVLA / RynnVLA-002：在一个 autoregressive 模型里同时生成 action 和 future image。
- DreamZero：14B autoregressive video diffusion WAM，joint video-action prediction，可作为 zero-shot policy。
- LingBot-VA：causal video-action world model，用 KV cache 和真实 observation feedback 做闭环控制。
- Fast-WAM：保留训练时 video co-training，但推理时不显式生成未来。

#### 2.1.3 从 2D 到 3D/4D

2D RGB 视频对具身来说不够，因为机器人需要：

- 尺度；
- 遮挡关系；
- 可达性；
- 接触；
- 稳定性；
- 物体 6D pose；
- deformable object 状态；
- 多视角一致性。

典型工作：

- TesserAct：生成 RGB、depth、normal videos，并重建 4D scene。
- WristWorld：用 4D world model 从第三人称视角生成 wrist-view。
- 4D-VLA：引入 RGB-D 与时序信息缓解坐标系统混乱。
- GigaWorld-0-3D：结合 3D generation、3DGS reconstruction、physical system identification、motion planning。

### 2.2 常用训练数据

#### 2.2.1 机器人轨迹数据

格式通常为：

```text
observation: RGB / RGB-D / multi-view
language instruction
robot proprioception
action
success / failure / reward, optional
```

常见数据源：

- Open X-Embodiment / RT-X
- RT-1
- DROID
- BridgeData
- CALVIN
- LIBERO
- RoboCasa / RoboCasa365
- RLBench
- ManiSkill
- RoboTwin
- AgiBot-World
- UMI data
- in-house robot trajectories

#### 2.2.2 人类视频

用途：

- 学习物理交互；
- 学习手-物体关系；
- 学习长时域操作流程；
- 扩展场景、物体和技能覆盖。

代表工作：

- DreamDojo：44k 小时 egocentric human videos。
- Pi0.7：使用 egocentric human videos 和 web auxiliary data。
- VPP、DreamGen、PhysicalAgent 也受益于人类视频或通用视频模型。

#### 2.2.3 通用互联网视频

用途：

- 学习时空视觉先验；
- 学习自然物理动态；
- 支撑 video diffusion / video DiT backbone。

代表 backbone：

- Cosmos-Predict2.5
- Wan2.2
- CogVideoX
- PaliGemma / Gemma / SigLIP 等 VLM/VLA 组件

#### 2.2.4 多视角、RGB-D、3D/4D 数据

用途：

- 训练多视角一致；
- 学习深度、normal、point cloud、occupancy；
- 支撑 3D/4D scene reconstruction。

代表：

- TesserAct 扩展 RGB-DN 数据；
- WristWorld 处理 anchor view 到 wrist view；
- 4D-VLA 使用 DROID、LIBERO/RLDS 等处理为 RGB-D；
- Embody4D/X-WAM/Kinema4D 等使用多视角、深度、点云或仿真 4D 数据。

#### 2.2.5 失败数据与 near-success 数据

越来越重要，原因是：

- 只用 expert demonstration 不知道如何恢复；
- world model 需要学习失败后果；
- VLA 需要知道什么动作会导致失败。

代表：

- Pi0.7 使用 autonomous evaluation data、suboptimal data、failure episodes，并用 quality/speed/mistake metadata 区分。
- World-VLA-Loop 使用 near-success / failure rollouts 改善 world model 与 VLA 的闭环。

### 2.3 仍未解决的问题

1. **动作可控性不足**：视频看起来合理，但未必严格服从 action。
2. **长时域误差累积**：rollout 越长越漂移。
3. **2D 指标与任务成功脱节**：FVD、LPIPS 好，不代表机器人成功率高。
4. **可消费输出不足**：生成 RGB 视频不一定能直接给 planner 或 VLA 使用。
5. **3D/4D 数据难获得**：多视角、深度、contact、force、deformable 数据成本高。
6. **接触与稳定性难建模**：堆叠、插入、折叠、软体操作仍困难。
7. **sim-to-real / generated-to-real gap**：合成数据和真实机器人行为之间仍有差距。
8. **inference latency**：test-time imagination 很贵，Fast-WAM 正是针对这一点提出质疑。

---

## 3. 问题 2：world model 训练完成后用来做什么

world model 训练完成后，主要有以下用途。

### 3.1 未来预测

给定当前观测和候选动作，预测未来：

- RGB video；
- multi-view video；
- depth / normal；
- object pose；
- contact / stability；
- success probability。

### 3.2 Model Predictive Control / Planning

流程：

```text
current observation
  -> sample candidate action sequences
  -> world model rollout
  -> reward / verifier scoring
  -> choose best action
  -> execute first chunk
  -> observe again
```

适合：

- stack；
- fold；
- drawer；
- tool use；
- insertion；
- long-horizon manipulation。

### 3.3 策略训练

world model 可以作为虚拟环境或辅助训练信号：

- model-based RL；
- policy optimization；
- synthetic rollout；
- failure recovery data generation。

代表：

- WMPO
- World-VLA-Loop
- DreamDojo downstream planning
- Cosmos Policy

### 3.4 VLA 辅助训练

VLA 不只学习 action，也学习动作后果：

```text
L = L_action + λ L_world
```

代表：

- WorldVLA
- RynnVLA-002
- Fast-WAM
- UVA
- UWM

### 3.5 数据增广 / data engine

world model 生成：

- 新视角；
- 新背景；
- 新物体；
- 新轨迹；
- wrist view；
- failure/near-success；
- counterfactual rollouts。

代表：

- GigaWorld-0
- WristWorld
- Cosmos
- DreamGen
- DreamDojo

### 3.6 策略验证与安全过滤

执行动作前，world model 预演：

- 是否碰撞；
- 是否掉落；
- 是否接触错误；
- 是否偏离目标；
- 是否不稳定。

这对真实机器人非常重要，因为真实试错成本高。

---

## 4. 问题 3：2D、3D、4D world model 的差异

| 类型 | 表征 | 训练数据 | 训练目标 | 优势 | 局限 | 适合场景 |
|---|---|---|---|---|---|---|
| 2D world model | RGB image/video tokens | 互联网视频、机器人视频、language-action-video | future frame/video prediction、action-conditioned video generation | 数据最多，模型成熟，可扩展 | 缺少显式几何、尺度、接触 | 数据增广、视觉预测、VLA 辅助训练 |
| 3D world model | depth、point cloud、occupancy、3DGS、mesh、object pose | RGB-D、多视角、仿真、重建数据 | 3D completion、novel view、object state prediction | 对机器人更可用，可处理遮挡和空间关系 | 数据贵，训练复杂，实时性差 | grasp、place、stack、navigation |
| 4D world model | dynamic 3D scene over time | 多视角视频、RGB-D video、4D reconstruction、action trajectories | 预测 3D 场景随动作演化 | 最接近具身需求，可表达运动、接触、时序 | 数据最难，评估最难，误差累积严重 | 操作、折叠、堆叠、接触、长时域任务 |

### 4.1 训练流程差异

#### 2D

```text
video/image tokenizer
  -> diffusion / AR / flow matching
  -> RGB future prediction
```

#### 3D

```text
multi-view / RGB-D
  -> depth / point cloud / 3DGS / occupancy
  -> 3D state prediction or reconstruction
```

#### 4D

```text
multi-view/RGB-D/action history
  -> dynamic 3D state
  -> future 3D/4D scene rollout
```

### 4.2 应用场景差异

- 2D：更适合视觉先验、数据增广、subgoal image。
- 3D：更适合抓取、摆放、避障、物体位姿。
- 4D：更适合动作后果、接触、稳定性、deformable object、长时域任务。

---

## 5. 问题 4：如何设计具身 world model 输出

对具身任务，world model 不应该只输出 RGB 视频。建议分层设计。

### 5.1 视觉层输出

- future RGB video；
- multi-view video；
- wrist-view video；
- depth video；
- normal video；
- segmentation mask；
- optical flow / scene flow。

作用：

- 给 VLA 作为视觉 prompt；
- 给 human/verifier 查看；
- 做 synthetic data；
- 做视频级 future imagination。

### 5.2 几何层输出

- object 6D pose；
- point cloud；
- mesh；
- 3D Gaussian；
- occupancy；
- gripper-object relative pose；
- occluded object belief；
- reachable region。

作用：

- 给 motion planner；
- 给 grasp planner；
- 做 collision checking；
- 做空间关系判断。

### 5.3 物理/交互层输出

- contact point；
- contact timing；
- support relation；
- stability score；
- slip/drop probability；
- force/tactile estimate；
- deformation state。

作用：

- 堆叠是否稳定；
- 插入是否成功；
- 布料是否折好；
- 是否需要重抓或修正。

### 5.4 任务层输出

- subgoal state；
- task predicate；
- progress score；
- success probability；
- reward；
- failure mode；
- uncertainty。

作用：

- reranking；
- MPC；
- failure recovery；
- task-level verifier。

### 5.5 推荐输出格式

针对 3-view embodied data，建议输出：

```text
Input:
  3-view RGB/RGB-D history
  + language instruction
  + candidate action chunk
  + robot proprioception

Output:
  future multi-view RGB-D / normal
  + object-centric 3D/4D state graph
  + contact / stability / affordance map
  + success / reward / uncertainty
```

---

## 6. 问题 5：3-view 输入下，怎样的终态表征才算任务解决

严格说，不能只靠终态 RGB 图像判断任务 100% 解决。需要一个可验证的任务状态表示：

```text
S_T = {
  calibrated 3D scene,
  object identities,
  object 6D poses or deformable mesh state,
  robot joint state / gripper state,
  contact graph,
  support relation,
  task predicate,
  stability score,
  uncertainty
}
```

### 6.1 刚体任务：叠箱子 / 堆积木

终态应包含：

- 每个箱子的 6D pose；
- stack order；
- 接触面；
- 重心是否落在支撑面内；
- 是否静止稳定；
- robot 是否释放；
- 是否无碰撞；
- 多视角一致性；
- 置信度。

成功谓词示例：

```text
box_A on box_B
AND pose_error < threshold
AND tilt_angle < threshold
AND center_of_mass_inside_support_polygon
AND stable for N seconds
AND no collision
AND gripper released
```

### 6.2 非刚体任务：叠衣服 / 毛巾折叠

终态应包含：

- cloth mesh / particle graph；
- corner/keypoint positions；
- fold line / crease；
- overlap area；
- wrinkle / flatness score；
- target shape similarity；
- cloth 是否静止；
- gripper 是否释放。

成功谓词示例：

```text
corner_alignment_error < threshold
AND folded_area_overlap > threshold
AND wrinkle_score < threshold
AND cloth_stable_for_N_frames
AND gripper_released
```

### 6.3 更合理的研究表述

论文中不建议声称“100% 解决”，更合理是：

- benchmark success rate 接近 95%-99%；
- OOD view/object/layout 下稳定；
- world model prediction 与真实成功高度相关；
- 下游 VLA/策略成功率显著提升；
- 真实机器人实验中表现稳定。

---

## 7. 问题 6：大小脑结构 VLA 中 world model 的角色

可以抽象为：

```text
大脑: VLM / LLM / VLA high-level reasoning
小脑: low-level controller / diffusion policy / action decoder
world model: imagination / prediction / verification / data engine
```

### 7.1 world model 的四种角色

#### 角色 1：想象器

```text
candidate action/subgoal -> future video/state
```

输出：

- future video；
- future 3D state；
- subgoal image；
- success probability。

代表：

- DreamZero；
- LingBot-VA；
- DreamDojo；
- PhysicalAgent。

#### 角色 2：验证器

```text
candidate rollout -> success / failure / safety score
```

用于：

- action reranking；
- collision filtering；
- stability checking；
- failure prediction。

#### 角色 3：数据生成器

```text
new scene / new view / new action -> synthetic trajectory
```

代表：

- GigaWorld-0；
- Cosmos；
- WristWorld；
- DreamGen；
- DreamDojo。

#### 角色 4：表征增强器

训练时通过 future prediction objective 学习 dynamics-aware representation，推理时不一定生成未来。

代表：

- Fast-WAM；
- UVA；
- VPP；
- UWM。

### 7.2 数据/表征增强的优势

| 增强 | 优势 |
|---|---|
| single image -> 2D video | 获得时间信息和动作后果 |
| 2D video -> multi-view video | 降低视角偏差，增强空间一致性 |
| 2D -> RGB-D / 3D | 获得尺度、遮挡、可达性和空间关系 |
| 3D -> 4D | 获得物体随动作演化的动态模型 |
| vision -> force/tactile | 支撑接触丰富任务，如插拔、压合、折叠 |

### 7.3 Pi0.7 的启发

Pi0.7 的主体不是显式 WAM，而是 VLA policy。world model 主要作为 **subgoal image generator**：

```text
high-level policy -> subtask instruction
world model -> subgoal image
Pi0.7 VLA -> action chunk
```

这说明 world model 不一定要直接控制机器人，它可以输出：

- visual waypoint；
- subgoal image；
- task-conditioned desired state。

对 3-view embodied task，可以进一步升级为：

```text
multi-view subgoal
+ 3D object state
+ contact/stability predicate
```

---

## 8. 问题 7：研究实验如何设计

建议研究命题：

> 面向具身 VLA 的 3-view action-conditioned 4D world state model：从多视角输入预测可验证的 object-centric future state，并用于策略选择、数据增广与失败恢复。

### 8.1 world model 本体评估

#### 视觉质量

- FVD；
- LPIPS；
- SSIM；
- PSNR；
- VBench；
- Q-Align。

#### 几何质量

- depth error；
- Chamfer distance；
- 6D pose error；
- point cloud consistency；
- multi-view consistency；
- reprojection error。

#### 动作对齐

- action-conditioned prediction error；
- counterfactual consistency；
- 同一初始状态下不同 action 是否导致不同合理结果；
- 预测未来与真实执行结果的相关性。

#### 物理一致性

- contact prediction accuracy；
- collision rate；
- support relation accuracy；
- stability prediction；
- slip/drop prediction。

#### 任务相关性

- predicted success vs real success correlation；
- reward calibration；
- failure mode classification；
- progress score correlation。

### 8.2 下游策略评估

建议对比：

| 设置 | 目的 |
|---|---|
| VLA baseline | 无 world model 的基线 |
| VLA + 2D video WM | 看纯视频预测是否有用 |
| VLA + subgoal image WM | 对齐 Pi0.7/Goal-VLA 思路 |
| VLA + latent-only WM | 对齐 Fast-WAM/UVA/VPP |
| VLA + 3D state WM | 看显式几何是否有增益 |
| VLA + 4D/contact WM | 证明 4D 与接触预测优势 |
| open-loop | 对比闭环方法 |
| closed-loop reranking/replanning | 证明 world model 的在线作用 |

### 8.3 必须做的 ablation

1. no world model；
2. 2D video world model；
3. 2D subgoal image；
4. latent future representation；
5. 3D object-centric state；
6. 4D object/contact/stability state；
7. training-only world modeling；
8. test-time explicit rollout；
9. single-view vs 3-view；
10. RGB vs RGB-D；
11. no failure data vs with failure/near-success data；
12. no uncertainty vs with uncertainty。

### 8.4 具身实验落地

#### 阶段 1：仿真

可选平台：

- LIBERO；
- RLBench；
- ManiSkill；
- RoboTwin；
- RoboCasa；
- CALVIN。

目标：

- 快速验证 world model 输出是否改善策略；
- 做大规模 ablation；
- 评估 OOD layout/object/view。

#### 阶段 2：真实机器人小规模

选 1-2 个任务，每个任务采集少量 3-view demonstrations：

- box/block stacking；
- towel/cloth folding。

测试：

- ID success；
- OOD object；
- OOD view；
- OOD layout；
- failure recovery。

#### 阶段 3：闭环

实现：

```text
observation
  -> VLA proposes candidate action chunks
  -> world model predicts structured future state
  -> verifier scores success/stability/contact
  -> execute best chunk
  -> observe again
```

---

## 9. 问题 8：推荐具体任务

建议选择 **一个刚体任务 + 一个非刚体任务**。

### 9.1 首选任务 1：box/block stacking

推荐原因：

- 成功状态容易定义；
- 3D 几何、接触、稳定性非常重要；
- 仿真和真实都容易做；
- 很适合证明 2D -> 3D/4D 的增益；
- LIBERO、RLBench、ManiSkill、RoboTwin、RoboCasa、DROID、BridgeData 中都有相似任务。

适合验证：

- object pose prediction；
- support relation；
- stability score；
- action reranking；
- multi-view consistency。

### 9.2 首选任务 2：cloth/towel folding

推荐原因：

- deformable object 能体现 4D world model 价值；
- 纯 VLA 或普通 policy 容易失败；
- 有相关数据与论文；
- 终态可以通过 keypoint、mesh、fold line、overlap、wrinkle 定义。

相关数据/工作：

- Flat'n'Fold；
- FoldNet；
- SSFold；
- Fast-WAM real-world towel folding；
- Pi0.7 laundry folding；
- PhysicalAgent rope/cloth-like tasks。

### 9.3 次选任务

| 任务 | 优点 | 风险 |
|---|---|---|
| drawer open/close | 数据多，评估清晰 | 几何挑战较弱 |
| cup/bowl stacking | 比 box 更真实 | 稳定性和 shape 多样性更难 |
| peg/plug insertion | contact-rich，研究价值高 | 需要 force/tactile 或高精度硬件 |
| rope/cable manipulation | 4D/deformable 价值高 | 数据与评估难 |
| table wiping | 可结合 contact/coverage | 成功定义需设计 |

推荐路线：

```text
阶段 1：box/block stacking
阶段 2：towel/cloth folding
阶段 3：扩展到 contact-rich 或 tool-use task
```

---

## 10. 重点论文与项目：链接、开源状态、backbone、架构和贡献

### 10.1 VLA / Pi 系列

| 工作 | 链接 | 开源状态 | backbone / 架构 | 做了什么 |
|---|---|---|---|---|
| Pi0 | https://arxiv.org/html/2410.24164v1 | 未完整开源 | PaliGemma 3B VLM + 300M action expert；conditional flow matching；50-step action chunk | 提出 flow-based VLA，支持高频连续动作和多机器人数据训练 |
| Pi0.5 | https://www.pi.website/download/pi05.pdf / https://proceedings.mlr.press/v305/black25a.html | 未完整开源 | 延续 Pi0 flow VLA，强化 heterogeneous task co-training | 面向 open-world generalization，在新家庭/新环境执行长时域任务 |
| Pi0.6 / Pi0.6-MEM | https://www.pi.website/ | 未完整开源 | Gemma3/SigLIP 类 VLM，action expert，memory/history encoder | 引入 embodied memory，增强长时域上下文 |
| Pi0.7 | https://arxiv.org/html/2604.15483v1 / https://www.pi.website/blog/pi07 | 未完整开源 | 继承 Pi0.6-MEM；up to 4 views、6 history frames、3 subgoal images；860M action expert；flow matching | 用 language、metadata、control mode、subgoal image 进行 steerable VLA 训练，使用 lightweight world model 生成 visual subgoal |

### 10.2 World Action Model / WAM

| 工作 | 链接 | 开源状态 | backbone / 架构 | 做了什么 |
|---|---|---|---|---|
| Fast-WAM | https://arxiv.org/html/2603.16666 / https://github.com/yuantianyuan01/FastWAM | 代码、权重、数据资源开放 | Wan2.2-5B video DiT + T5 + video VAE + 1B action expert；MoT shared attention；structured mask | 证明 WAM 主要收益来自训练时 video co-training，而非推理时显式生成未来 |
| DreamZero | https://arxiv.org/html/2602.15922 / https://github.com/dreamzero0/dreamzero | 代码、权重、推理与 benchmark 代码开放 | 14B autoregressive video diffusion backbone；joint video-action flow matching；KV cache | WAM as zero-shot policy，联合生成未来视频和动作，7Hz 闭环控制 |
| LingBot-VA | https://arxiv.org/html/2601.21998v2 / https://github.com/robbyant/lingbot-va | 代码和模型开放 | Wan2.2-5B video stream + action stream；dual-stream MoT；causal mask；Noisy History Augmentation | causal world modeling for robot control，强调闭环、长记忆和异步推理 |
| Motus | https://arxiv.org/abs/2512.13030 / https://github.com/thu-ml/Motus | 代码开放 | Wan2.2-5B + Qwen3-VL-2B + action expert + understanding expert；MoT；latent action | 统一 latent action world model，支持 WAM/VLA/IDM/video generation 多模式 |
| WorldVLA | https://arxiv.org/html/2506.21539v1 / https://github.com/alibaba-damo-academy/WorldVLA | 代码开放 | Chameleon-style autoregressive multimodal model；image/action/text tokenizers | 统一 action generation 和 future image prediction |
| RynnVLA-002 | https://arxiv.org/abs/2511.17502 / https://github.com/alibaba-damo-academy/RynnVLA-002 | 代码和权重开放 | VLA + world model 统一框架；continuous action transformer、wrist camera、state input | WorldVLA 后续，在 LIBERO 和真实 LeRobot 上提升明显 |
| UVA | https://arxiv.org/html/2503.00200v1 / https://unified-video-action-model.github.io/ | 项目开放 | VAE kl-f16 + Transformer + decoupled video/action diffusion heads | 联合 video-action latent，训练时预测视频和动作，推理可跳过视频生成 |
| VPP | https://arxiv.org/abs/2412.14803 / https://github.com/roboterax/video-prediction-policy | 代码开放 | text-guided video prediction model + predictive visual representation aggregation | 使用 video diffusion 内部 predictive representation 训练机器人策略 |

### 10.3 Foundation robot world model / data engine

| 工作 | 链接 | 开源状态 | backbone / 架构 | 做了什么 |
|---|---|---|---|---|
| Cosmos | https://arxiv.org/html/2501.03575v1 / https://github.com/NVIDIA/Cosmos | 代码、tokenizer、模型权重开放；训练数据不完全开放 | diffusion / autoregressive world foundation models；video tokenizer；flow matching | 面向 Physical AI 的 world foundation model 平台 |
| Cosmos-Predict2.5 | https://github.com/nvidia-cosmos/cosmos-predict2.5 | 代码和权重开放 | flow-based Video2World / Text2World / Image2World；2B/14B | 预测未来世界状态，可做机器人 post-training |
| DreamDojo | https://arxiv.org/html/2602.06949 / https://github.com/NVIDIA/DreamDojo | 代码、模型、Hugging Face 资源开放 | Cosmos-Predict2.5 + WAN2.2 tokenizer + DiT；700M latent action model；distilled causal student | 用 44k 小时人类第一视角视频预训练 robot world model，post-train 到目标机器人 |
| DreamGen | https://proceedings.mlr.press/v305/jang25a.html | 与 Cosmos 生态相关资源开放 | video world model + latent action / inverse dynamics | 用视频世界模型生成机器人数据，提升泛化 |
| GigaWorld-0 | https://giga-world-0.github.io/ / https://github.com/open-gigaai/giga-world-0 | GitHub 和 Hugging Face 模型开放，Apache 2.0 | GigaWorld-0-Video + GigaWorld-0-3D；sparse attention、FP8、3DGS、system identification | 把 world model 作为 VLA data engine |
| GigaBrain-0 | https://github.com/open-gigaai/giga-brain-0 | 代码开放 | VLA trained on GigaWorld-0 generated data | 验证 world-model-generated data 对真实机器人性能的提升 |
| Vid2World | https://knightnemo.github.io/vid2world/ | 项目页提供代码/模型资源 | causalized video diffusion；causal action guidance | 把 passive video diffusion model 转成 interactive world model |

### 10.4 3D / 4D world model

| 工作 | 链接 | 开源状态 | backbone / 架构 | 做了什么 |
|---|---|---|---|---|
| TesserAct | https://tesseractworld.github.io/ / https://github.com/UMass-Embodied-AGI/TesserAct | 代码、权重、LoRA fine-tune 支持开放 | CogVideoX-based video generation + RGB-DN learning + 4D reconstruction | 输入图像和文本，生成 RGB/depth/normal video，再重建 4D scene |
| WristWorld | https://arxiv.org/abs/2510.07313 / https://github.com/XuWuLingYu/WristWorld | 推理代码和权重开放；训练代码未完全开放 | VGGT extension + wrist head + SPC loss + DiT generation | 从第三人称 anchor view 生成 wrist-view video |
| 4D-VLA | https://github.com/fudan-zvg/4D-VLA | 代码开放，MIT | RGB-D spatiotemporal VLA；DROID pretraining；LIBERO evaluation | 用 RGB-D 和时序信息缓解 VLA 坐标混乱 |
| Embody4D | https://arxiv.org/abs/2605.01799 | 以论文为主 | Wan2.1-T2V-1.3B + warp-then-inpaint + adaptive noise + interaction-aware attention | 单目具身视频到任意新视角 4D 视频生成 |
| X-WAM | https://arxiv.org/abs/2604.26694 | 以论文为主 | unified 4D world-action model；多视角 RGB-D、异步 denoising | 面向 4D world-action modeling |
| Kinema4D | https://arxiv.org/abs/2603.12746 | 以论文为主 | URDF/kinematic robot 4D pointmap + RGB/pointmap future | 动作条件机器人 4D 仿真 |
| RoboStereo | https://arxiv.org/abs/2603.12639 | 以论文为主 | RGB tower + XYZ pointmap tower + 4D Gaussian head | 4D embodied world model 与策略优化 |
| MVISTA-4D | https://arxiv.org/abs/2603.07039 | 以论文为主 | single-view RGB-D -> multi-view future RGB-D + inverse action | imagine-then-act manipulation |

### 10.5 VLA + world model reasoning / subgoal / force

| 工作 | 链接 | 开源状态 | backbone / 架构 | 做了什么 |
|---|---|---|---|---|
| Hume | https://arxiv.org/html/2505.21432v4 / https://github.com/hume-vla/hume | 代码和权重开放 | dual-system VLA；System-2 value-guided thinking + System-1 reactive control | 用低频思考和高频控制提升机器人任务表现 |
| TriVLA | https://zhenyangliu.github.io/TriVLA/ / https://arxiv.org/html/2507.01424 | 项目页显示代码和数据 | System 1 policy + System 2 VLM + System 3 video diffusion world model | 三系统 VLA，引入 episodic world modeling |
| Goal-VLA | https://nus-lins-lab.github.io/goalvlaweb/ / https://arxiv.org/abs/2506.23919 | 代码 Coming Soon | image-generative VLM as object-centric world model | 生成 goal image，再做 object transformation 和 zero-shot manipulation |
| DreamVLA | https://zhangwenyao1.github.io/DreamVLA/ / https://github.com/Zhangwenyao1/DreamVLA | 代码和权重开放 | dynamic region + depth + DINOv2/SAM semantic features + inverse dynamics | 用综合世界知识预测增强 VLA |
| PhysicalAgent | https://arxiv.org/abs/2509.13903 | 论文称开放系统/接口/评测，但未确认明确 GitHub | VLM reasoning + image-to-video foundation model + lightweight video-to-control adapter | 生成候选动作视频，闭环执行与失败纠正 |
| ForceVLA | https://arxiv.org/abs/2505.22159 / https://sites.google.com/view/forcevla2025 | 代码和数据计划释放 | force-aware MoE fusion；vision/proprioception/force | 把 6 轴力觉作为 VLA 一等模态，处理 contact-rich manipulation |

### 10.6 综述与资源

| 资源 | 链接 | 开源状态 | 内容 |
|---|---|---|---|
| 3D and 4D World Modeling: A Survey | https://arxiv.org/html/2509.07996v1 / https://github.com/worldbench/survey | survey repo 开放 | 系统梳理 3D/4D world modeling、数据和评测 |
| A Comprehensive Survey on World Models for Embodied AI | https://arxiv.org/html/2510.16732v1 / https://github.com/Li-Zn-H/AwesomeWorldModels | bibliography 开放 | 从功能、时间建模、空间表征三个维度梳理 embodied world models |
| Awesome World Models | https://github.com/knightnemo/Awesome-World-Models | 开放 | world model 论文列表 |
| Awesome World Models for Robots | https://github.com/operator22th/awesome-world-models-for-robots | 开放 | robot world model 论文列表 |

---

## 11. 关键架构对比：world model backbone 与设计选择

### 11.1 Backbone 类型

| Backbone 类型 | 代表工作 | 优势 | 问题 |
|---|---|---|---|
| VLM backbone + action expert | Pi0、Pi0.5、Pi0.7、OpenVLA | 语义强，推理快，易做 instruction following | 缺少显式 dynamics/world modeling |
| video diffusion / video DiT backbone | Cosmos、Wan2.2、DreamZero、Fast-WAM、LingBot-VA | 时空物理先验强，适合未来预测 | 推理慢，action alignment 难 |
| joint video-action Transformer | WorldVLA、UVA、Motus | 统一动作和视觉建模 | token/attention 设计复杂 |
| 3D/4D reconstruction/generation backbone | TesserAct、WristWorld、4D-VLA | 几何友好，适合具身 | 数据和算力成本高 |
| latent action model | DreamDojo、Motus | 可从无动作视频学习可迁移交互表示 | latent action 到机器人 action 的对齐难 |

### 11.2 推理时是否显式生成未来

| 路线 | 代表工作 | 推理形式 | 适合 |
|---|---|---|---|
| 显式 test-time imagination | DreamZero、LingBot-VA、DreamDojo、WorldVLA | 生成 future video/state + action | planning、verification、可解释控制 |
| 训练时 world modeling，推理时直接 action | Fast-WAM、UVA、VPP | 用 world latent/action head，不解码视频 | 低延迟真实部署 |
| world model 生成 subgoal，再给 VLA | Pi0.7、Goal-VLA、PhysicalAgent | subgoal image/video/state -> VLA | high-level guidance、长时域任务 |
| data engine | GigaWorld-0、Cosmos、DreamGen | 离线生成训练数据 | 数据扩充、OOD 泛化 |

### 11.3 Attention / token 设计

| 工作 | token 设计 | attention / mask |
|---|---|---|
| Pi0.7 | observation image tokens、subgoal image tokens、text tokens、state tokens、action tokens | image/subgoal 内部 bidirectional；text causal；action expert bidirectional attend VLM activations |
| Fast-WAM | clean first-frame latent、noisy future video latent、action tokens | action tokens 不 attend future video；推理移除 future video branch |
| DreamZero | visual latent、language、state、action latent | autoregressive chunk-wise video；joint video-action denoising；KV cache |
| LingBot-VA | interleaved video/action tokens | causal AR sequence；shared attention；Noisy History Augmentation |
| UVA | history visual/action tokens、masked future visual tokens、action chunks | Transformer joint latent；decoupled video/action diffusion heads |
| DreamDojo | video latent + chunked latent/robot actions | action chunk 注入对应 latent frame；distilled causal student |

---

## 12. 推荐研究方案

### 12.1 推荐研究题目

```text
3-view Action-Conditioned 4D World State Model for Embodied VLA
```

或中文：

```text
面向具身 VLA 的三视角动作条件 4D 世界状态模型
```

### 12.2 核心主张

相比只生成 2D future video 或 subgoal image，面向具身任务的 world model 应输出：

```text
future multi-view RGB-D
+ object-centric 3D/4D state
+ contact/stability/reward/uncertainty
```

并证明这些结构化输出能提升：

- 策略成功率；
- OOD 泛化；
- failure recovery；
- 数据效率；
- 真实机器人安全性。

### 12.3 推荐系统框架

```text
3-view RGB/RGB-D history
+ language
+ proprioception
+ candidate action chunks
        ↓
action-conditioned 4D world model
        ↓
future multi-view RGB-D
+ object-centric 3D/4D graph
+ contact/stability/success/uncertainty
        ↓
verifier / reward model / planner
        ↓
select action chunk
        ↓
execute and observe
```

### 12.4 与已有工作的差异化

| 对比对象 | 你的潜在优势 |
|---|---|
| Pi0.7 visual subgoal | 你的输出不仅是 2D subgoal image，而是 multi-view + 3D/4D + 可验证 predicate |
| Fast-WAM / UVA | 不只训练时 world modeling，还研究 structured future state 在 test-time reranking 中的价值 |
| DreamZero / LingBot-VA | 更关注 3-view embodied task 和几何/contact 可验证性，可能更低成本 |
| DreamDojo | 可借鉴 latent action，但聚焦特定任务下 3D/4D 结构化输出 |
| TesserAct / WristWorld | 不只生成 4D/视角，而是服务 VLA/policy success |

### 12.5 最小可行实验

任务：

1. box/block stacking；
2. towel/cloth folding。

输入：

- single-view RGB；
- 3-view RGB；
- 3-view RGB-D。

输出对比：

- 2D future video；
- subgoal image；
- latent representation；
- 3D object state；
- 4D object/contact state。

使用方式：

- training-only auxiliary loss；
- test-time action reranking；
- failure recovery；
- synthetic data augmentation。

核心指标：

- task success rate；
- OOD success；
- predicted success vs real success correlation；
- action controllability；
- 3D pose/depth/contact/stability accuracy；
- inference latency。

---

## 13. 最后建议

如果目标是形成清晰、有竞争力的研究方案，不建议从“做一个通用巨大 world model”开始。更建议聚焦：

> **具体任务 + 多视角输入 + 结构化 3D/4D 输出 + VLA/策略闭环收益。**

最推荐的切入：

1. **刚体稳定性任务：box/block stacking**
   - 用来证明 3D pose、contact、support、stability 的价值。

2. **非刚体动态任务：towel/cloth folding**
   - 用来证明 4D deformable state、fold line、keypoint、mesh/particle graph 的价值。

最终论文贡献可以围绕一个核心问题展开：

> 对 embodied VLA 来说，world model 的价值到底来自训练时动态表征、推理时未来想象，还是结构化 3D/4D 可验证状态？

如果能通过系统实验回答这个问题，会比单纯提出一个新的视频生成模型更有研究价值。
