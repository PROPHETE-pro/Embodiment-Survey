# 4D World Model 论文精读整理

> 覆盖论文：arXiv:2605.01799、2604.26694、2604.16733、2604.21915、2603.16669、2603.12746、2603.12639、2603.07039、2603.01549、2602.09878、2601.05138、2512.03000。
>
> 整理目标：逐篇回答“解决什么问题、如何训练、创新点、训练数据、输入输出”，并汇总这些文章涉及的数据集。

## 0. 总览

这 12 篇文章虽然都围绕“4D world model”，但其实分成几类不同问题：

1. **具身机器人世界模型 / world-action model**
   - Embody4D、X-WAM、Kinema4D、RoboStereo、Pri4R、MVISTA-4D。
   - 共同关注：机器人操作中的时空几何、动作条件未来预测、策略验证或策略学习。
2. **动态视频/场景的 4D 生成与重拍摄**
   - Vista4D、VerseCrafter。
   - 共同关注：给定源图像/视频和相机/物体轨迹，生成视角一致、时间一致的动态视频。
3. **主动感知、评测与数据构建**
   - AW4RE、Thinking in Dynamics/Dyn-Bench、DynamicVerse。
   - 共同关注：如何在动态 4D 世界中感知、检索证据、评估多模态大模型，或大规模构建 4D 标注。
4. **地球尺度 4D 时空世界模型**
   - DeepEarth/Earth4D。
   - 关注点从机器人/视频转向地球观测：用 4D 时空位置编码支撑多模态自监督世界模型。

可以用下表快速定位每篇论文的“4D”含义：

| 论文 | 4D 表示/能力 | 主要任务 |
|---|---|---|
| Embody4D | 单目具身视频到任意新视角视频 | embodied novel-view video generation |
| X-WAM | 多视角 RGB-D 未来 + 3D 重建 + 动作 | unified world-action model |
| AW4RE | 历史观测检索形成局部 3D/4D evidence，再补全反事实视角 | 主动感知与 counterfactual sensing |
| Vista4D | 单目动态视频 lifting 成 4D point cloud，用于 video reshooting | 动态视频重拍摄 |
| Kinema4D | URDF/运动学机器人 4D pointmap + RGB/pointmap 未来 | 动作条件 4D 机器人仿真 |
| Thinking in Dynamics | 4D 动态理解 benchmark，而非生成模型 | MLLM 动态推理/grounding 评测 |
| RoboStereo | RGB tower + XYZ pointmap tower + 4D Gaussian head | 4D EWM 与策略优化 |
| DeepEarth/Earth4D | lat/lon/elevation/time 的 4D hash embedding | 地球观测多模态世界模型 |
| Pri4R | 训练时用 3D point tracks 作为 privileged 4D supervision | VLA 世界动态表征增强 |
| MVISTA-4D | 单视角 RGB-D 到多视角未来 RGB-D，并反推 action | imagine-then-act manipulation |
| VerseCrafter | 背景点云 + 物体 3D Gaussian trajectories | 4D 几何控制视频生成 |
| DynamicVerse | 从单目视频自动生成 metric point maps、camera、mask、caption | 大规模 4D 数据集/标注框架 |

---

## 1. Embody4D: A Generalist 4D World Model for Embodied AI

- 链接：https://arxiv.org/abs/2605.01799
- 机构：浙江大学、北京中关村学院、中国科学技术大学、中科院自动化所、北航等。

### 1.1 目标解决什么问题？

Embody4D 目标是把具身机器人视频世界模型从 **2D 单视角视频预测** 提升到 **4D 多视角动态生成**：给定一个单目机器人操作视频，合成同一动态过程在任意目标相机视角下的视频。

论文指出三个瓶颈：

1. **成对多视角具身动态数据稀缺**：真实机器人数据常只有头部/腕部相机，缺少全局 arbitrary view。
2. **新视角生成的几何/时序不一致**：warp-then-inpaint 类方法在遮挡、边界、低置信区域易产生伪影。
3. **操作细节幻觉**：机械臂、夹爪、接触点、被操作物体的细粒度动态容易被生成模型“补错”。

### 1.2 模型训练如何设计？

基础模型为 **Wan2.1-T2V-1.3B**，采用 flow matching 视频生成框架，整体是 **warp-then-inpaint**：

1. 从源视频重建深度/相机/点云。
2. 将源点云投影到目标相机，得到 warped RGB、geometry/occupancy mask。
3. 用轻量 confidence module 估计区域置信度。
4. 根据置信度执行 adaptive noise injection。
5. 用带 interaction-aware attention 的视频 DiT 生成目标视角视频。

训练分两类数据：

- **23K 合成 4D 样本**：MuJoCo Menagerie 里 30 种机器人/机械臂形态与 DL3DV 背景合成，使用 VGGT 重建背景相机和深度，双虚拟相机保证源/目标视角一致。
- **24K 真实单目具身数据**：来自 AGIBOT、Rh20t、Robset、BC-Z、Interndata-A1，用于学习真实操作和交互区域。

核心损失：

- 主损失是 flow matching velocity regression。
- 辅助损失约束 confidence map 同时接近几何 mask 和 Wan encoder 特征差异。
- 自适应噪声策略对不同置信区域使用不同噪声强度：高置信区域保纹理/结构，低置信区域保留生成自由度。

### 1.3 创新点

1. 面向具身操作的单目到任意视角 **4D video-to-video world model**。
2. 用 MuJoCo Menagerie + DL3DV 构建跨形态机器人合成 4D 数据。
3. **confidence-aware adaptive noise injection**：让高/低置信区域在 flow matching 中采用不同噪声调度。
4. **interaction-aware attention**：用前景/交互 mask bias 强化机械臂与被操作物体区域。
5. 论文进一步用生成数据微调 VLA policy，验证其下游机器人规划/学习价值。

### 1.4 使用什么数据训练？

| 数据 | 用途 | 规模/说明 |
|---|---|---|
| MuJoCo Menagerie | 机器人前景合成 | 30 种机器人/机械臂形态 |
| DL3DV | 多视角真实背景 | GPT-4o 过滤、VGGT 重建相机/深度 |
| AGIBOT | 真实单目机器人数据 | 真实操作学习 |
| Rh20t | 真实机器人操作数据 | 真实操作学习 |
| Robset | 真实具身数据 | 真实操作学习 |
| BC-Z | 机器人操作数据 | 真实操作学习 |
| Interndata-A1 | 真实具身操作数据 | 真实操作学习 |

### 1.5 模型输入输出

- 输入：单目机器人操作视频、目标相机视角/轨迹；内部还使用源视频点云投影、warped RGB、occupancy mask、interaction mask。
- 输出：目标视角下的具身操作视频；论文实验中为 49 帧、384x672。

### 1.6 实验结论

评测集为 120 个单目视频：70 个合成样本、50 个真实开源样本。指标包括 VBench 和 Q-Align。Embody4D 在 Subject、Background、Temporal、Motion、Imaging、Q-Align Visual Quality 上均优于 ReCamMaster、Ex-4D、Reangle-A-Video、TrajectoryCrafter。典型结果：

| 方法 | Subject | Background | Temporal | Motion | Imaging | Q-Align |
|---|---:|---:|---:|---:|---:|---:|
| TrajectoryCrafter | 0.9202 | 0.9388 | 0.9714 | 0.9911 | 0.6257 | 3.8954 |
| Embody4D | **0.9477** | **0.9408** | **0.9751** | **0.9945** | **0.6994** | **3.9970** |

---

## 2. X-WAM: Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising

- 链接：https://arxiv.org/abs/2604.26694
- 机构：清华大学、小米机器人、北京大学、中科院自动化所。

### 2.1 目标解决什么问题？

X-WAM 目标是把 **高质量未来世界生成、3D/4D 空间重建、机器人动作实时执行** 统一到一个模型里。它批评已有 unified world/action model 多停留在 2D pixel-space，缺少显式空间信息；同时视频生成需要多步 denoising，而动作解码要求低延迟，二者节奏冲突。

### 2.2 模型训练如何设计？

基础模型是 **Wan2.2-5B** 视频扩散/DiT 先验。输入多视角 RGB 观测、机器人 proprioception、noisy action tokens 等，联合输出未来多视角 RGB-D 视频、机器人状态与动作。

两个关键训练/结构设计：

1. **Lightweight depth adaptation**
   - 不把 depth 作为额外通道直接拼接，避免 token 长度接近翻倍。
   - 复制预训练 DiT 最后若干 block，形成 depth prediction branch。
   - depth branch 读取 RGB 主干特征，保护 RGB 视频先验。
   - depth branch 推理时可关闭以降低策略延迟。

2. **Asynchronous Noise Sampling (ANS)**
   - 低维动作可用少数 denoising steps 恢复，高质量视频需要完整去噪。
   - 推理时先用较少初始 steps 解码动作并执行，再继续去噪生成高质量视频。
   - 训练时从匹配推理过程的 joint timestep distribution 采样，避免训练/推理 mismatch。

论文还提到真实任务中用 DCT loss 平滑动作。

### 2.3 创新点

1. 统一 4D world generation、RGB-D/3D reconstruction 和 action execution。
2. 轻量 depth branch 在不显著增加 token 长度的情况下引入空间监督。
3. ANS 同时兼顾实时动作解码与视频生成质量。
4. 明确证明 depth/spatial supervision 对 policy success rate 也有增益。

### 2.4 使用什么数据训练？

- **超过 5,800 小时机器人数据**预训练。可读正文/摘要中未列出完整组成。
- 评测数据/基准：
  - RoboCasa。
  - RoboTwin 2.0。
  - 真实机器人 earphone packing 任务。

### 2.5 模型输入输出

- 输入：多视角 RGB observation、机器人 proprioceptive state、noisy action tokens、任务上下文/指令。
- 输出：
  - future multi-view RGB videos；
  - future depth maps；
  - RGB-D 重建出的 3D/4D point clouds；
  - future robot states/actions。

### 2.6 实验结论

- RoboCasa 平均成功率：**79.2%**。
- RoboTwin 2.0 平均成功率：**90.7%**。
- 论文称视觉和几何指标均超过已有方法，并展示真实机器人耳机打包任务。

---

## 3. AW4RE: Active World-Model with 4D-informed Retrieval for Exploration and Awareness

- 链接：https://arxiv.org/abs/2604.16733
- 机构：KavAI、UCSD。

### 3.1 目标解决什么问题？

AW4RE 面向 **physical awareness**：智能体在大型动态环境中应如何选择传感器/相机动作，以减少对真实 4D 世界状态的不确定性。

它把问题形式化为有限时域 spatio-temporal POMDP：

- 状态：完整 4D 时空环境。
- 动作：相机位置、朝向、zoom、内参/外参等 sensing action。
- 观测：该相机动作下获得的 RGB 视频。
- 奖励：任务奖励 + 信息增益 - sensing cost。

核心目标不是普通视频预测，而是构造一个可查询的 surrogate environment：给定历史部分观测和一个反事实相机动作，预测“如果这样观察会看到什么”。

### 3.2 模型训练/系统设计如何设计？

AW4RE 更像 **检索 + 几何投影 + 条件生成补全** 的系统，而非端到端重新训练的单一世界模型。它把 action-conditioned observation model 分成三步：

1. **4D-informed state estimator**
   - 不维护全局 4D reconstruction。
   - 对每个查询帧，按时间和相机 action 从历史库中检索最多 M 个相关帧。
   - 同一时间帧可用所有像素；跨时间帧只使用静态区域，动态区域 mask 掉。
   - 得到局部 3D proxy / point cloud。

2. **Evidence-backed observation module**
   - 把局部 3D proxy 投影到查询相机。
   - 几何支持稀疏时做 action-conditioned densification。
   - 输出 evidence-supported partial frame。

3. **Conditional generative observation module**
   - 用条件视频扩散模型补全 unsupported 区域。
   - 实验使用 Cosmos1.0 Diffusion 7B Video2World 的 GEN3C fine-tuned 版本。

### 3.3 创新点

1. 将 physical awareness 建模为相机 sensing POMDP。
2. 不是只生成固定未来，而是支持反事实 sensing action 查询。
3. 4D-informed retrieval：按时间、视角和动态/静态区域选择证据。
4. 显式区分 evidence-backed 和 hallucinated/completed 区域，减少无依据幻觉。

### 3.4 使用什么数据？

- **Waymo Open Dataset**：反事实相机查询评测。
- GEN3C / Cosmos1.0 Diffusion 7B Video2World：作为条件生成模块/基线。

### 3.5 模型输入输出

- 输入：历史 RGB 视频、每帧相机 action/内外参、新查询相机 action 序列。
- 输出：查询 action 下的预测 RGB 视频；中间结果包括 retrieved local point clouds、partial evidence frames。

### 3.6 实验结论

在 partial temporal evidence 场景中，AW4RE 明显优于 GEN3C：

| 查询 | 方法 | Full PSNR | Full SSIM | Full LPIPS | Evidence PSNR | Evidence SSIM | Evidence LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Next time steps | AW4RE | 28.532 | 0.905 | 0.100 | 31.864 | 0.927 | 0.057 |
| Next time steps | GEN3C | 15.269 | 0.467 | 0.602 | 13.520 | 0.411 | 0.626 |
| Previous time steps | AW4RE | 32.265 | 0.927 | 0.058 | 32.768 | 0.919 | 0.049 |
| Previous time steps | GEN3C | 26.677 | 0.780 | 0.211 | 26.111 | 0.717 | 0.233 |

时间一致性 T-LPIPS 上 AW4RE 也显著更低。

---

## 4. Vista4D: Video Reshooting with 4D Point Clouds

- 链接：https://arxiv.org/abs/2604.21915
- 机构：Eyeline Labs、Netflix、Columbia University、UCLA、Stony Brook University、Oxford 等。

### 4.1 目标解决什么问题？

Vista4D 研究 **video reshooting**：给定真实动态视频和用户指定的新相机轨迹，重新合成同一动态场景在新视角/新轨迹下的视频。

挑战包括：

1. 单目动态视频的 4D reconstruction/depth 有伪影。
2. 只依赖点云条件会丢失源视频外观。
3. 很难精确跟随复杂 target camera。
4. 动态对象和静态背景在新视角下容易闪烁或畸变。

### 4.2 模型训练如何设计？

Vista4D 基于 **Wan2.1-T2V-14B** flow matching 视频 DiT。

核心流程：

1. **temporally-persistent 4D point cloud**
   - 用 4D reconstruction 得到源视频 depth、camera intrinsics/extrinsics。
   - 用 segmentation 得到 static pixel mask。
   - 每帧 RGB-D lift 到 world-space point cloud。
   - 静态像素在所有帧中 persistent，从而在目标相机下提供更强的 seen-content 约束。

2. **训练时使用 noisy reconstructed multiview data**
   - 不是只用干净 frontal depth。
   - 使用多视角动态视频经过 4D reconstruction 后的 noisy point cloud render，使训练分布接近真实推理时的几何伪影。

3. **in-context conditioning**
   - 条件包括源视频、source point cloud 在目标相机下的 render、alpha mask、target camera。
   - 将源视频 latent、point cloud render latent、noisy target latent 沿 frame 维拼接，而非只用 cross-attention 注入。
   - 相机用 Plucker embeddings，经 zero-initialized projection 注入。

训练细节：

- 672x384 训练 30,000 steps，再 1280x720 训练 300 steps。
- 49 帧，global batch size 8，AdamW，lr=1e-5。
- 训练 patchify layers、self-attention、camera encoders/projectors，其余冻结。

### 4.3 创新点

1. 让静态像素跨时间持久化，形成更稳定的 4D point cloud prior。
2. 用 noisy multiview 4D reconstruction 训练，提高真实推理鲁棒性。
3. 同时条件于源视频和点云渲染，兼顾外观保真与相机控制。
4. 支持 video reshooting 之外的 dynamic scene expansion、4D scene recomposition、long video inference with memory。

### 4.4 使用什么数据训练？

| 数据 | 用途 |
|---|---|
| MultiCamVideo | 合成多视角同步动态视频；用 STream3R 做 4D reconstruction |
| OpenVidHD-0.4M | 取 60K 真实单目视频子集；用 pi^3 做 4D reconstruction |
| DAVIS + Pexels | 构建 51 个视频、110 个 video-camera pair 的评测集 |
| iphone dataset | 真实多视角同步 novel-view video synthesis 评测 |

### 4.5 模型输入输出

- 输入：源视频、用户目标相机轨迹、4D point cloud 渲染到目标相机得到的 guidance、alpha mask。
- 输出：同一动态场景在目标相机轨迹下的视频。

### 4.6 实验结论

Camera control 和 3D consistency：

| 方法 | Translation ↓ | Rotation ↓ | Intrinsics ↓ | RE@SG ↓ |
|---|---:|---:|---:|---:|
| GEN3C | 1.309 | 4.751 | 5.085 | 12.99 |
| Vista4D | **1.251** | **4.647** | **4.927** | **7.504** |

iphone novel-view synthesis 中 Vista4D 在 mPSNR、mLPIPS、PSNR、LPIPS、EPE 上表现最好或接近最好。用户研究中，Vista4D 在源内容保留、相机准确性、总体保真度上分别获得 67.06%、68.17%、77.38% 偏好率，远高于 baselines。

---

## 5. Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation

- 链接：https://arxiv.org/abs/2603.16669
- 机构：NTU S-Lab、CUHKSZ。

### 5.1 目标解决什么问题？

Kinema4D 目标是构建动作条件的 4D 生成式机器人仿真器。它认为机器人-世界交互不是 2D 视频事件，而是 **3D 空间 + 时间 + 精确机器人运动约束** 的 4D 事件。

现有生成式机器人仿真常见问题：

- 只在 RGB 2D 空间生成。
- 用语言或 latent action token 控制，缺少精确运动学约束。
- 难以模拟接触、遮挡、变形、near-miss 和失败执行。

### 5.2 模型训练如何设计？

Kinema4D 将任务解耦为：

1. **Kinematic Control**
   - 标准机器人使用 CAD/URDF。
   - 未知机器人可通过环绕视频、Grounded-SAM2/SAM2、ReconViaGen 重建 textured mesh，再与 URDF joint anchors 对齐。
   - 末端位姿序列经 IK 得到关节角；关节空间动作直接映射或积分。
   - 每时刻 FK 得到各 link 6DoF pose。
   - 投影到主视角，生成 robot pointmap；每个像素存相机坐标系下的 x,y,z。

2. **4D Generative Modeling**
   - 基于 Wan2.1 14B + 4DNex 4D-aware 权重，使用 LoRA 微调。
   - 输入初始 RGB、robot pointmap 序列、robot occupancy mask。
   - 输出同步 RGB 序列与 world pointmap 序列。
   - RGB/pointmap 共享 VAE 和 RoPE，用 domain embedding 区分模态。
   - 用 diffusion denoising MSE 训练。

### 5.3 创新点

1. 机器人动作不交给生成模型猜，而是通过 URDF + IK/FK 转成确定的 4D robot trajectory。
2. 用 pixel-aligned robot pointmap 作为时空控制信号。
3. 生成模型专注于环境反应，形成“确定性机器人控制 + 生成式环境动力学”的解耦。
4. 同时生成 RGB 与 pointmap，使结果具有视觉和几何一致性。
5. 构建 Robo4D-200k，大规模 4D 机器人交互数据。

### 5.4 使用什么数据训练？

- **Robo4D-200k**：201,426 个 real + synthetic 机器人交互 episode。
- 来源：
  - DROID。
  - Bridge/BridgeData。
  - RT-1。
  - LIBERO。
- 真实视频用 ST-V2 生成 4D pointmap 伪标注；LIBERO 使用原生 depth。
- 每个 episode 统一采样为 49 帧；2% 作为 stratified validation benchmark。

### 5.5 模型输入输出

- 输入：初始 RGB 图像、动作序列、机器人 URDF/几何、相机参数、由运动学生成的 robot pointmap。
- 输出：未来 RGB 视频、未来 world pointmap/4D 几何序列。

### 5.6 实验结论

视频生成指标：

| 方法 | PSNR ↑ | SSIM ↑ | Latent L2 ↓ | FID ↓ | FVD ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ctrl-World | 21.03 | 0.803 | 0.1533 | 24.9 | 112.8 | 0.122 |
| TesserAct | 19.35 | 0.766 | 0.1911 | 29.5 | 120.3 | 0.158 |
| Kinema4D | **22.50** | **0.864** | **0.1380** | 25.2 | **98.5** | **0.105** |

几何指标对比 TesserAct：

| 方法 | CD-L1 ↓ | CD-L2 ↓ | F-Score ↑ | F-Score temporal ↑ |
|---|---:|---:|---:|---:|
| TesserAct | 0.0836 | 0.0130 | 0.2896 | 0.9523 |
| Kinema4D | **0.0479** | **0.0077** | **0.4733** | **0.9686** |

---

## 6. Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World

- 链接：https://arxiv.org/abs/2603.12746
- 核心产物：Dyn-Bench。

### 6.1 目标解决什么问题？

这篇论文不是训练一个新的生成式 4D world model，而是评测 MLLM 是否真的能在物理 4D 世界中“thinking in dynamics”：

- 感知动态物体。
- 跟踪时空变化。
- 理解物体间交互。
- 在相机运动下推理深度、方向、事件顺序。
- 同时完成语言 VQA 和动态对象 grounding。

### 6.2 模型训练/评测设计如何设计？

主要贡献是构建 **Dyn-Bench** benchmark：

- 1K videos。
- 7K VQA pairs。
- 3K dynamic object grounding pairs。

任务分为三层：

1. **Dynamic Inter-Object Perception**：接近、遮挡、追赶、超越、相对运动等。
2. **Dynamic Object-Scene Tracking**：进入/离开场景、状态变化、场景关系变化。
3. **Dynamic Camera-Object Reasoning**：相机运动下的深度、方向、相对位置和事件顺序。

论文还提出 **ST-TCM: Spatio-Temporal Textual Cognitive Map**：

- 从 RGB-D、segmentation mask 重建 3D object trajectories。
- 提取对象位置、尺寸、朝向、object-object 与 camera-object 关系。
- 转换为结构化文本，辅助问题生成和模型推理。

视觉增强策略包括：

- Masked Frames Only。
- Mask-Guided Fusion：融合原始帧与 mask，引导模型关注动态区域。

### 6.3 创新点

1. 将 4D 动态理解拆成 VQA 和 dynamic object grounding 两条能力线。
2. 同时覆盖 2D 视频分割数据和 4D 动态场景数据。
3. ST-TCM 把时序语义、运动动态、空间几何转成结构化文本认知图。
4. 发现常规 CoT/caption prompt 增益有限，mask-guided fusion 和 ST-TCM 更有效。

### 6.4 使用什么数据？

Dyn-Bench 来源包括：

- 2D 视频分割数据：DAVIS、SA-V、DynPose-100K、YouTube-VIS。
- 4D 动态场景数据：DynamicReplica、PointOdyssey、Spring、Total-Recon。

### 6.5 模型输入输出

- VQA 输入：视频、多选问题，可选 ST-TCM 文本或 mask-guided video。
- VQA 输出：多选答案。
- Grounding 输入：视频和动态对象指代表达/问题。
- Grounding 输出：对应对象的视频分割 mask。

### 6.6 实验结论

论文评测 general、spatial、region-level MLLM。主要发现：

- 现有 MLLM 难以同时做好 spatio-temporal reasoning 和 dynamic object grounding。
- Qwen3-VL 系列在 VQA 上较强，Sa2VA 系列在 grounding 上更强。
- 完整 ST-TCM 明显提高 VQA；以 Qwen3-VL-32B 为例，Avg 从约 62.8 提升到 68.3。
- Mask-Guided Fusion 比只叠 mask 更有效；Qwen3-VL-8B Avg 从 53.8 提升到 57.1。

---

## 7. RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization

- 链接：https://arxiv.org/abs/2603.12639
- 机构：清华大学、X Square Robot、HKUST。

### 7.1 目标解决什么问题？

RoboStereo 面向 embodied world model 的两个核心缺陷：

1. **几何幻觉**：object teleportation、scale drift、surface penetration 等。
2. **缺少统一策略优化框架**：已有工作零散处理 test-time verification、training-time refinement 或 open exploration。

目标是构建高保真 4D mental simulator，并用它统一支持 policy optimization。

### 7.2 模型训练如何设计？

架构是 **对称双塔 DiT**：

- RGB video tower。
- XYZ pointmap tower。
- 两塔都基于 Cosmos 2.5 backbone。
- 双向 cross-attention：
  - RGB tower 从 pointmap tower 获取几何约束。
  - pointmap tower 从 RGB tower 获取语义上下文。
- 4D Gaussian Splatting head 将 RGB-XYZ 序列转为 dynamic Gaussian Splats，支持 flexible viewpoint rendering。

动作是 7D：

`(dx, dy, dz, dtheta_r, dtheta_p, dtheta_y, gripper_width)`

动作注入为 dual-path action-conditioned timestep embedding：

- MLP1 与 diffusion timestep embedding 融合。
- MLP2 预测 AdaLN modulation offsets。
- 在每个 denoising step 提供 frame-level structural guidance。

训练两阶段：

1. **Independent Training**
   - RGB tower 和 pointmap tower 分别从 Cosmos 2.5 初始化并独立 fine-tune。
   - Rectified Flow velocity matching。
   - 条件包括初始帧、text instruction、action sequence。

2. **Joint Training**
   - 集成两个 tower。
   - 冻结 FFN，只微调 self-attention 和 cross-attention。
   - RGB/pointmap paired data，共享 diffusion timestep，独立 noise。
   - 总损失：`lambda_v L_v + lambda_p L_p`，其中 `lambda_v=0.85`，`lambda_p=0.15`。

策略优化框架包括：

- **TTPA**：test-time imagined rollout + video understanding model 打分，安全/成功才执行。
- **IEPL**：专家动作和 policy 动作分别在 RoboStereo rollout，用 LPIPS 作为视觉 imitation reward，用 GRPO 更新。
- **OEPL**：无专家示范，在 RoboStereo 闭环探索，用 reward model 对 clip completion confidence 打分，再 GRPO。

### 7.3 创新点

1. RGB/XYZ 双塔对称建模，而非单向几何辅助。
2. 双向 cross-attention 让视觉语义和几何结构互相增强。
3. 4D Gaussian head 支持多视角监督和 reward。
4. 首个较完整地把 4D world model 用于 test-time、imitation、open exploration 三类 VLA policy optimization 的框架。

### 7.4 使用什么数据训练？

- **Bridge V2**：约 20,000 个第三视角机器人厨房操作视频。
- 视频标准化为 320x256、5 FPS。
- 每帧配 7D gripper action。
- 使用 **DepthAnything V3** 预测 depth，再反投影为 XYZ pointmaps。

### 7.5 模型输入输出

- 输入：初始 RGB、初始 XYZ pointmap、文本任务、7D 动作序列。
- 输出：未来 RGB video、未来 XYZ pointmap video、4D Gaussian representation、imagined rollouts。

### 7.6 实验结论

摘要级结论：

- RoboStereo 达到 state-of-the-art generation quality。
- 统一策略优化框架在 fine-grained manipulation tasks 上带来 **超过 97% 平均相对提升**。
- 双塔和 cross-modal enhancement 减少 object teleportation、scale drift、surface penetration。

可读正文中没有完整核对到所有逐任务成功率和生成指标表，因此建议把 “>97% relative improvement” 作为论文摘要报告结论，而非额外推断。

---

## 8. DeepEarth / Earth4D: Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding

- 链接：https://arxiv.org/abs/2603.07039

### 8.1 目标解决什么问题？

DeepEarth 目标是构建地球观测领域的 self-supervised multi-modal world model，并提出一个可在全球尺度、长时间跨度上工作的 4D space-time positional encoder：**Earth4D**。

问题背景：

- 地球观测数据跨模态、跨空间尺度、跨多年/世纪。
- 模型需要把 latitude、longitude、elevation、time 与图像/文本/传感器数据融合。
- 直接 4D dense grid 不可行，内存爆炸。

### 8.2 模型训练如何设计？

DeepEarth：

- modality-specific encoders 编码图像、文本、sensor data 等。
- Earth4D 将 `(latitude, longitude, elevation, time)` 映射为 learnable positional embedding。
- 多模态 token + Earth4D token 进入 autoencoder context window。
- 自监督目标是 masked multi-modal reconstruction，学习联合分布并支持模拟/重建。

Earth4D：

- 扩展 NVIDIA multi-resolution hash encoding 到 4D。
- 采用 Grid4D 式分解：
  - xyz 空间网格。
  - xyt、yzt、xzt 三个时空网格。
- 4 个 grid 并行计算并拼接。
- 默认 24 个 resolution levels，每 level 最多 2^22 entries，每 entry 2D feature，输出 192D。
- 使用 learned hash probing 减少 hash collision。

LFMC 实验：

- Earth4D 编码 `(x,y,z,t)` 为 192D。
- 拼接 learnable species embedding。
- MLP 输出 Live Fuel Moisture Content 百分比。

### 8.3 创新点

1. 提出 planetary-scale 4D space-time positional encoder。
2. 用 xyz/xyt/yzt/xzt 分解避免完整 4D 网格爆炸。
3. learned hash probing 显著降低 collision 并提升预测性能。
4. 在 Globe-LFMC 2.0 上，仅用坐标+物种 embedding 就超过使用遥感/天气/地形等多模态输入的 Galileo baseline。

### 8.4 使用什么数据训练/评测？

- **Globe-LFMC 2.0**：Live Fuel Moisture Content 预测 benchmark。
  - train：76,467。
  - test：13,297。
- Galileo baseline 使用：
  - Sentinel-2 optical imagery。
  - Sentinel-1 SAR。
  - VIIRS night lights。
  - ERA5 weather。
  - TerraClimate。
  - SRTM topography。
  - species type、coordinates/time。

### 8.5 模型输入输出

- 通用 DeepEarth 输入：多模态地球观测数据 + 4D 坐标。
- 通用输出：masked modality reconstruction、多模态联合表示。
- LFMC 实验输入：`(x,y,z,t)` + species name/embedding。
- LFMC 输出：LFMC 百分比。

### 8.6 实验结论

| 模型 | 输入 | MAE ↓ | RMSE ↓ | R2 ↑ |
|---|---|---:|---:|---:|
| Galileo | 坐标 + 物种 + 遥感/天气/地形等 | 12.6 pp | 18.9 pp | 0.72 |
| Earth4D + learned hashing | 坐标 + species name | **11.7 pp** | **18.7 pp** | **0.783** |

learned probing 相比普通 hash encoding：

- RMSE 26.0 -> 18.7。
- MAE 16.6 -> 11.7。
- R2 0.58 -> 0.783。

---

## 9. Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation

- 链接：https://arxiv.org/abs/2603.01549
- 机构：KAIST AI、LG AI Research、Yonsei、SNU、CMU。

### 9.1 目标解决什么问题？

Pri4R 关注 VLA 模型的一个弱点：行为克隆动作标签只告诉模型“怎么动”，但不告诉模型“世界会怎样响应动作”。因此 VLA 可能语义理解正确，却缺少物理/几何动态理解。

Pri4R 的目标是：训练时用 **3D point tracks** 作为 privileged 4D supervision，让 VLA 学到动作-世界动态关系；推理时移除辅助头，不增加任何输入/输出/计算开销。

### 9.2 模型训练如何设计？

在 VLA 上增加轻量 point track head：

- 当前点集 `P_t` 经 PointMLP 编码。
- VLA backbone 动作相关特征 `z_t` 与点特征融合。
- FusionMLP 预测未来每步 3D displacement。

接入方式：

- OpenVLA-OFT：使用 final-layer action-query token hidden states 作为 `z_t`。
- pi0/pi0.5：用轻量 transformer embedding module 从 VLM 最终层图像/语言 token 得到 horizon-level embedding。

损失：

- 保留原 VLA action loss：
  - OpenVLA-OFT：连续动作 L1 regression。
  - pi 系列：flow matching action objective。
- 加入 point-track L1 loss：
  - `L = L_act + omega_pt * L1(predicted point displacement, GT displacement)`。
- 默认 `omega_pt=1`，每条轨迹采样 `N_p=1024` 点。

3D point tracks 构造：

- 仿真：利用 simulator GT mesh，采样表面点并用 face index + barycentric coordinate 跟踪。
- 真实数据：用 off-the-shelf 3D point tracking model 生成伪标签，并用 segmentation 让采样更关注机器人和物体前景。

### 9.3 创新点

1. 3D point tracks 作为训练时 privileged supervision，而非推理输入。
2. 3D point tracks 同时具备时序密集、metric 3D、空间稀疏、与动作同空间等优点。
3. 与 OpenVLA-OFT、pi0、pi0.5 等不同 VLA 架构兼容。
4. 推理时移除 point-track head，保持原 VLA 架构和延迟。

### 9.4 使用什么数据训练？

- **LIBERO**：Spatial、Object、Goal、Long 四个 suite；每 suite 10 个任务，每任务 50 demos。
- **RoboCasa Human-50**：24 个 kitchen manipulation atomic tasks，每任务 50 demos。
- **真实机器人数据**：用于真实任务评测；包括 obstacle pick-place、bin placing、farthest object、moving object 等任务。
- OpenVLA/OFT 背后涉及 OpenX 预训练，但 Pri4R 主实验训练/评测聚焦 LIBERO、RoboCasa 和真实机器人数据。

### 9.5 模型输入输出

- 训练/推理输入：语言指令、多视角 RGB、机器人 proprioception。
- 策略输出：动作 chunk。
- 训练时额外输出：未来 3D point displacements。
- 推理时额外输出：无。

### 9.6 实验结论

LIBERO：

| 模型 | Baseline Avg | +Pri4R Avg |
|---|---:|---:|
| pi0 | 87.4 | 90.6 |
| pi0.5 | 92.6 | 94.0 |
| OpenVLA-OFT | 92.7 | **96.3** |

OpenVLA-OFT 在 LIBERO-Long：85.5 -> **95.3**。

RoboCasa：

| 模型 | Baseline Avg | +Pri4R Avg |
|---|---:|---:|
| pi0 | 38.8 | 42.2 |
| pi0.5 | 52.9 | 57.0 |
| OpenVLA-OFT | 33.1 | **46.3** |

消融结论：

- 3D tracks 优于 2D tracks。
- 时间密集轨迹优于稀疏目标点。
- robot + environment tracks 优于只跟踪机器人或只跟踪环境。
- depth 监督有帮助，但不如 point tracks，因为 depth 缺少跨时序点身份一致性。

---

## 10. MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation

- 链接：https://arxiv.org/abs/2602.09878

### 10.1 目标解决什么问题？

MVISTA-4D 研究机器人 manipulation 中的 **imagine-then-act**：

1. 先用世界模型从单视角 RGB-D + 指令生成多视角未来 RGB-D。
2. 再从生成的 4D future 中推断可执行动作。

它要解决两个问题：

- 单视角或纯 RGB future prediction 几何不完整、不一致。
- 从 imagined future 反推动作时，inverse dynamics 是病态问题：多个动作可能导致相似视觉变化。

### 10.2 模型训练如何设计？

生成模型基于 latent video diffusion / flow matching，配合 3D VAE tokenizer 和 DiT-style backbone。

关键设计：

1. **输入组织**
   - 同一视角内 RGB-D 用 width-wise concatenation，让 appearance/geometry token 邻近。
   - 不同视角用 height-wise concatenation，促进跨视角结构级信息交换。

2. **跨模态融合**
   - learnable modality token 区分 RGB/depth。
   - local cross-modality attention 在局部窗口交换信息。
   - gated residual update 控制信息注入。

3. **跨视角几何一致性**
   - 相机 embedding 用围绕 shared look-at point 的球坐标：yaw、pitch、roll Fourier features + log distance，共 13D。
   - geometry-aware deformable cross-view attention 沿 epipolar line 采样 candidate key/value，再预测 offset refinement。

4. **动作轨迹 latent**
   - 用 TCN-based VAE 把整段动作序列编码为低维 latent tokens。
   - latent 作为 style code 通过 cross-attention 注入生成模型。
   - 训练时加 latent-consistency head，从最终 hidden tokens 重构 trajectory latent，避免模型忽略动作条件。

5. **测试时动作推断**
   - 先用文本/初始观测生成 future rollout。
   - 冻结 rollout，随机初始化 trajectory latent，通过反传优化使 `G(l,z)` 复现该 future。
   - 用 TCN decoder 解码 action prior。
   - residual inverse dynamics model 输入连续 3D point sets 和 prior action，输出 correction。

### 10.3 创新点

1. 从单视角 RGB-D 生成多视角未来 RGB-D，可回投影融合为更完整 4D point-cloud dynamics。
2. 同时显式建模 RGB-depth 一致性和多视角几何一致性。
3. 把整段动作轨迹压缩成 latent style code，避免逐帧动作条件的脆弱时间对齐。
4. 用测试时 latent optimization + residual IDM 替代直接 inverse dynamics。

### 10.4 使用什么数据训练？

- **RLBench**。
- **RoboTwin**。
- 作者采集的真实机器人多视角数据集：14 个任务。

### 10.5 模型输入输出

- 生成阶段输入：单视角 RGB-D observation、文本指令、目标相机 extrinsics；训练时还可输入 action trajectory latent。
- 生成阶段输出：reference view 和目标 views 的未来 RGB-D sequence，可回投影为动态点云。
- 动作阶段输出：可执行 action trajectory。

### 10.6 实验结论

论文在 RLBench、RoboTwin 和真实机器人多视角数据集上评估 4D generation 与 manipulation。可读摘要/正文确认其在下游 manipulation 上超过强基线；外部可读摘录显示约 RLBench 72.6%、RoboTwin 43.0% success rate。具体 generation metric 表格在可读片段中未完全核对，因此这里不把未核对数值展开。

---

## 11. VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control

- 链接：https://arxiv.org/abs/2601.05138
- 机构：复旦大学、Shanghai Innovation Institute、HKU、腾讯 ARC Lab。

### 11.1 目标解决什么问题？

VerseCrafter 目标是在真实视频生成中统一控制：

- camera motion；
- 多物体 3D motion；
- camera 与 object 的协同动态。

它认为 2D trajectories、masks、boxes 缺乏 3D awareness；3D boxes 太刚性；SMPL-X 等类别受限。因此需要一种统一、可编辑、类别无关的 4D 几何控制表示。

### 11.2 模型训练如何设计？

核心表示：**4D Geometric Control**

- 静态背景点云 `P_bg`。
- 每个可控物体一条 3D Gaussian trajectory：
  - mean 表示 3D motion path。
  - covariance 表示空间范围、朝向、近似形状。

构造控制信号：

1. 从输入图像用 MoGe-2 估计 depth/intrinsics。
2. 用 Grounded-SAM2 获取对象 masks。
3. 背景点云由非对象区域回投影得到。
4. 每个对象点云拟合 full-covariance 3D Gaussian。
5. 用户可在 Blender 等 3D 编辑器中拖动/keyframe ellipsoid。

渲染为 4D control maps：

- background RGB/depth。
- 3D Gaussian trajectory RGB/depth。
- soft merged mask。

生成模型：

- 冻结 Wan2.1 T2V-14B。
- 只训练轻量 **GeoAdapter**。
- control maps 经 Wan Encoder 编码，mask 插值到 latent resolution。
- GeoAdapter 与 Wan-DiT block 交错，每隔 `k=5` 个 DiT block 注入 residual modulation。
- 文本 prompt 由 umT5 编码，同时注入 Wan-DiT 和 GeoAdapter。

训练细节：

- Adam，lr=2e-5。
- 16 张 96GB GPU，global batch size 16。
- 两阶段：2500 iterations on 480p，2500 iterations on 720p。
- CFG text dropout 0.1；推理 50 denoising steps，CFG scale 5.0。

### 11.3 创新点

1. 提出统一的 4D Geometric Control，把相机和多物体运动放在共享世界坐标系。
2. 用 3D Gaussian trajectory 替代 3D box/sparse trajectory/类别特定人体模型。
3. frozen large video prior + lightweight GeoAdapter，兼顾质量与可控性。
4. 构建 VerseControl4D，解决真实视频缺少 4D 控制监督的问题。

### 11.4 使用什么数据训练？

- **VerseControl4D**
  - 来源：Sekai-Real-HQ、SpatialVID-HQ。
  - 35,000 training samples，1,000 validation samples。
  - 每 clip 81 frames。
  - 约 26% 来自 Sekai-Real-HQ，74% 来自 SpatialVID-HQ。
  - 约 20% training samples 是 static scenes，用于 camera-only world exploration。
- 标注工具：
  - Qwen2.5-VL-72B caption。
  - Grounded-SAM2 object masks。
  - MegaSAM + MoGe-2 + UniDepth V2 生成 depth/camera/geometry。

### 11.5 模型输入输出

- 输入：reference image、text prompt、camera trajectory、object masks、object 3D Gaussian trajectories。
- 输出：动态真实视频，支持 camera-only、object-only、joint camera-object control。

### 11.6 实验结论

Joint camera + object motion control：

| 方法 | Overall ↑ | RotErr ↓ | TransErr ↓ | ObjMC ↓ |
|---|---:|---:|---:|---:|
| Perception-as-Control | 83.66 | 5.006 | 8.767 | 6.556 |
| Yume | 85.47 | 7.560 | 8.735 | 7.959 |
| Uni3C | 83.55 | 1.361 | 7.731 | 5.883 |
| VerseCrafter | **88.10** | **0.890** | **3.103** | **2.507** |

Camera-only static scenes：

| 方法 | Overall ↑ | RotErr ↓ | TransErr ↓ |
|---|---:|---:|---:|
| FlashWorld | 85.33 | 1.792 | 3.257 |
| VerseCrafter | **86.80** | **0.650** | **2.587** |

消融显示：3D Gaussian trajectory 优于 3D bounding box 和 3D point trajectory；depth-aware rendering、background/foreground decoupled control 都重要。

---

## 12. DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling

- 链接：https://arxiv.org/abs/2512.03000
- 机构：XMU、CUHK、UT Austin、UW、PKU、Meta 等。

### 12.1 目标解决什么问题？

DynamicVerse 关注 4D world model 的数据瓶颈：现有 4D 数据常来自有限 simulator 或传统 SfM，存在：

- sim-to-real gap。
- up-to-scale geometry，缺少 metric scale。
- 缺少 dynamic object masks、object/camera/scene captions。
- 难以从互联网单目视频中规模化构建高质量 4D 数据。

目标是从 raw monocular videos 自动生成物理尺度、多模态 4D 标注。

### 12.2 模型训练/系统设计如何设计？

DynamicGen pipeline 两大阶段：

1. metric-scale geometry and moving object recovery。
2. hierarchical dynamic content caption generation。

主要步骤：

1. **4D scene curation**
   - 汇聚真实视频数据和已有 4D/synthetic 数据。

2. **data filtering**
   - 指标：proximal depth、focal-length stability、video blur、camera motion smoothness、non-perspective distortion。
   - 用约 1000 个人工标注视频训练 Random Forest 输出 0-5 质量分。
   - 再用 VLM 判断剔除不适合重建的视频。

3. **moving object recovery**
   - Qwen2.5-VL 识别 moving objects 和类别。
   - SA2VA 根据类别生成 masks。
   - 结合几何标注提取 physical size 和 3D bounding box。

4. **dynamic bundle adjustment**
   - 输出每帧 point map、camera intrinsics、camera pose。
   - 优化项包括 static BA reprojection、optical-flow consistency、non-rigid dynamic structure、dynamic motion regularization、camera smoothness。
   - 五阶段：dynamic masking、coarse camera initialization、static BA、non-rigid BA、sliding-window global refinement。

5. **dynamic content captioning**
   - object caption：DAM，输入 video + object masks。
   - scene caption：Qwen2.5-VL + hierarchical prompt。
   - camera caption：根据 inter-frame transformations 描述 pan/tilt/zoom/dolly。
   - LLM rephrasing 和 human-in-the-loop review。

### 12.3 创新点

1. 用 foundation models + dynamic BA 从 monocular web videos 自动生成 metric-scale 4D annotations。
2. 同时提供 geometry、camera、dynamic object masks/categories、captions。
3. caption 分 object/scene/camera 三粒度，适合 4D world understanding。
4. 大规模数据集 DynamicVerse：100K+ scenes/videos、800K+ masklets、10M+/13.6M frames。

### 12.4 使用什么数据？

DynamicVerse 来源：

- 真实 2D video datasets：DAVIS2017、YouTube-VIS、UVO-dense、VOST、BURST、MOSE、SA-V。
- 4D/synthetic/posed datasets：PointOdyssey、Spring、Dynamic Replica、MVS-Synth、RealCam-Vid、DynPose-100K。

评测：

- Sintel、KITTI：video depth。
- 论文还评估 camera pose estimation、camera intrinsics estimation；可读片段显示还涉及 TUM-dynamics 等。

### 12.5 输入输出

- 输入：raw monocular RGB videos。
- 输出：
  - metric-scale per-frame point maps/depth。
  - camera intrinsics/poses。
  - static/dynamic separation。
  - moving object masks/categories/3D boxes。
  - object-level、scene-level、camera-motion captions。

### 12.6 实验结论

DynamicGen 在 video depth、camera pose、camera intrinsics 三类 benchmark 上评估。可读表格显示在 Sintel/KITTI 上对比 MonST3R、Uni4D、Depth-pro、Metric3D、DepthCrafter 等；论文结论称 DynamicGen/DynamicVerse 在 metric-scale 和 global accuracy 上优于现有方法。caption quality 通过 GPT-assisted/G-VEval 类指标和 human study 验证。

---

## 13. 数据集整理对比

下表汇总这些论文涉及的训练集、评测集、数据来源和辅助基准。若一个数据集只是作为 baseline 所用外部输入或评测工具，会在“用途”中说明。

| 数据集/资源 | 类型/领域 | 涉及论文 | 用途 | 规模/标注特点 |
|---|---|---|---|---|
| MuJoCo Menagerie | 机器人模型库/仿真 | Embody4D | 合成机器人前景 | 30 种跨形态机器人/机械臂 |
| DL3DV | 多视角真实场景 | Embody4D | 合成 4D 背景、相机对 | 用 VGGT 重建深度/相机，用 GPT-4o 过滤 |
| AGIBOT | 机器人操作 | Embody4D | 真实单目具身数据 | 真实操作/交互区域学习 |
| Rh20t | 机器人操作 | Embody4D | 真实具身数据 | 真实机器人操作 |
| Robset | 机器人操作 | Embody4D | 真实具身数据 | 真实机器人操作 |
| BC-Z | 机器人操作 | Embody4D | 真实具身数据 | 机器人操作数据 |
| Interndata-A1 | 机器人操作 | Embody4D | 真实具身数据 | 真实具身操作 |
| 5,800h robotic data | 大规模机器人数据 | X-WAM | 预训练 unified 4D world-action model | 论文摘要报告超过 5,800 小时，组成未完全公开于可读片段 |
| RoboCasa | 机器人仿真/厨房操作 | X-WAM、Pri4R | 成功率评测/训练评测 | X-WAM 报 79.2%；Pri4R 用 RoboCasa Human-50 |
| RoboTwin / RoboTwin 2.0 | 机器人仿真/双臂操作 | X-WAM、MVISTA-4D | 操作成功率和 4D generation 评测 | X-WAM RoboTwin 2.0 报 90.7% |
| Waymo Open Dataset | 自动驾驶多相机视频 | AW4RE | 反事实相机查询评测 | 用于 viewpoint/scale/time gap 查询 |
| MultiCamVideo | 合成多视角动态视频 | Vista4D | 训练 video reshooting | 用 STream3R 做 4D reconstruction |
| OpenVidHD-0.4M | 真实单目视频 | Vista4D | 训练 | 取 60K 子集，用 pi^3 做 4D reconstruction |
| DAVIS / DAVIS2017 | 视频分割/真实动态视频 | Vista4D、Thinking in Dynamics、DynamicVerse | 评测/数据来源 | Vista4D 构建评测视频；Dyn-Bench/DynamicVerse 数据来源 |
| Pexels | stock videos | Vista4D | 评测集来源 | 与 DAVIS 组成 51 视频、110 video-camera pairs |
| iphone dataset | 多视角真实动态视频 | Vista4D | novel-view video synthesis 评测 | 有同步多视角 GT |
| Robo4D-200k | 4D 机器人交互 | Kinema4D | 训练/验证 | 201,426 episodes，RGB + 4D pointmap |
| DROID | 真实机器人示范 | Kinema4D | Robo4D-200k 来源 | 真实 RGB video，经 ST-V2 伪 4D 标注 |
| Bridge / BridgeData V2 | 机器人厨房操作 | Kinema4D、RoboStereo | Kinema4D 数据来源；RoboStereo 训练 | RoboStereo 用约 20K third-person videos |
| RT-1 | 机器人示范 | Kinema4D | Robo4D-200k 来源 | 真实机器人示范 |
| LIBERO | 机器人仿真 manipulation | Kinema4D、Pri4R | Kinema4D 合成成功/失败；Pri4R 训练/评测 | Pri4R 四个 suite，每任务 50 demos |
| Dyn-Bench | 4D 动态理解 benchmark | Thinking in Dynamics | MLLM VQA/grounding 评测 | 1K videos、7K VQA、3K grounding |
| SA-V | 视频分割/大规模视频 | Thinking in Dynamics、DynamicVerse | Dyn-Bench/DynamicVerse 来源 | DynamicVerse 表中 50.9K videos、4.2M frames、642.6K masklets |
| DynPose-100K | posed dynamic videos | Thinking in Dynamics、DynamicVerse | Dyn-Bench/DynamicVerse 来源 | 100K 级动态视频/pose 数据 |
| YouTube-VIS | 视频实例分割 | Thinking in Dynamics、DynamicVerse | Dyn-Bench/DynamicVerse 来源 | 视频实例 mask/category |
| DynamicReplica / Dynamic Replica | 4D 动态场景 | Thinking in Dynamics、DynamicVerse | Dyn-Bench/DynamicVerse 来源 | synthetic/realistic 4D scenes |
| PointOdyssey | synthetic 4D dynamic scenes | Thinking in Dynamics、DynamicVerse | Dyn-Bench/DynamicVerse 来源 | 4D trajectories/geometry |
| Spring | synthetic 4D dynamic scenes | Thinking in Dynamics、DynamicVerse | Dyn-Bench/DynamicVerse 来源 | 真实感 synthetic dynamic scenes |
| Total-Recon | 4D 动态场景 | Thinking in Dynamics | Dyn-Bench 来源 | 用于动态场景理解 |
| Bridge V2 + DepthAnything V3 pseudo-depth | 机器人操作 + 几何伪标注 | RoboStereo | RGB/XYZ 双塔训练 | 约 20K third-person videos，320x256，5 FPS，7D action |
| Globe-LFMC 2.0 | 生态/地球观测 | DeepEarth/Earth4D | LFMC 预测评测 | train 76,467，test 13,297 |
| Sentinel-2 | 遥感 optical | DeepEarth/Galileo baseline | Galileo baseline 输入 | 光学遥感 |
| Sentinel-1 | 遥感 SAR | DeepEarth/Galileo baseline | Galileo baseline 输入 | SAR |
| VIIRS night lights | 夜光遥感 | DeepEarth/Galileo baseline | Galileo baseline 输入 | 夜光 |
| ERA5 | 气象/再分析 | DeepEarth/Galileo baseline | Galileo baseline 输入 | weather/climate |
| TerraClimate | 气候/水文 | DeepEarth/Galileo baseline | Galileo baseline 输入 | soil/water climate |
| SRTM | 地形 | DeepEarth/Galileo baseline | Galileo baseline 输入 | topography |
| Pri4R real-robot data | 真实机器人 | Pri4R | 真实任务评测/伪 3D tracks | 固定相机，任务含 obstacle pick-place、bin placing 等 |
| RLBench | 机器人仿真 manipulation | MVISTA-4D | 4D generation + downstream manipulation | 多任务 manipulation benchmark |
| MVISTA real-robot multiview dataset | 真实机器人多视角 | MVISTA-4D | 真实平台评测 | 作者采集 14 个任务 |
| VerseControl4D | 真实视频 4D 控制 | VerseCrafter | 训练/验证 | 35K train、1K val，81 frames/clip |
| Sekai-Real-HQ | 世界探索真实视频 | VerseCrafter | VerseControl4D 来源 | 约占训练集 26% |
| SpatialVID-HQ | 空间视频数据 | VerseCrafter | VerseControl4D 来源 | 约占训练集 74% |
| DynamicVerse | 大规模 4D 多模态数据集 | DynamicVerse | 数据集产物/训练资源 | 100K+ videos/scenes、800K+ masklets、10M+/13.6M frames |
| UVO-dense | 视频对象分割 | DynamicVerse | 数据来源 | 1.0K videos、68.3K frames、10.2K masklets |
| VOST | 视频对象分割 | DynamicVerse | 数据来源 | 0.7K videos、75.5K frames |
| BURST | 视频对象分割 | DynamicVerse | 数据来源 | 2.9K videos、195.7K frames |
| MOSE | 视频对象分割 | DynamicVerse | 数据来源 | 2.1K videos、638.8K frames |
| MVS-Synth | synthetic multiview/4D | DynamicVerse | 数据来源 | synthetic outdoor/urban |
| RealCam-Vid | posed video/4D | DynamicVerse | 数据来源 | 100K 级 posed video |
| Sintel | synthetic video benchmark | DynamicVerse | video depth/camera 评测 | depth/pose/intrinsics benchmark |
| KITTI | 自动驾驶 benchmark | DynamicVerse | video depth 评测 | metric outdoor driving depth |
| TUM-dynamics | 动态 RGB-D/pose | DynamicVerse | camera pose 评测 | 可读片段提及，具体表格未完全核对 |
| VBench / VBench-I2V / VBench-2.0 | 视频生成评测套件 | Embody4D、Vista4D、VerseCrafter | 视频质量/一致性指标 | 不是训练数据集 |
| Q-Align | 视觉质量评测模型/基准 | Embody4D | 视觉质量评分 | 不是训练数据集 |

### 13.1 数据集按用途归类

**机器人操作训练/评测数据**：

- AGIBOT、Rh20t、Robset、BC-Z、Interndata-A1。
- DROID、Bridge/Bridge V2、RT-1、LIBERO。
- RoboCasa、RoboTwin/RoboTwin 2.0、RLBench。
- Pri4R/MVISTA 作者自采真实机器人数据。

这些数据主要服务于动作条件 world model、VLA 政策学习和操作成功率评测。

**视频/动态场景生成数据**：

- DL3DV、MultiCamVideo、OpenVidHD-0.4M、DAVIS、Pexels、iphone dataset。
- Sekai-Real-HQ、SpatialVID-HQ、VerseControl4D。

这些数据主要服务于 novel-view synthesis、video reshooting、camera/object control。

**4D/动态场景标注数据**：

- DynamicReplica、PointOdyssey、Spring、Total-Recon、MVS-Synth、RealCam-Vid、DynPose-100K。
- DynamicVerse。

这些数据主要用于构建或评估 4D 几何、相机、动态对象和时序理解能力。

**地球观测数据**：

- Globe-LFMC 2.0、Sentinel-1/2、VIIRS、ERA5、TerraClimate、SRTM。

这些数据不属于机器人/视频生成主线，但展示了 4D world model 在 planetary-scale space-time embedding 中的另一种形态。

### 13.2 横向比较结论

1. **机器人方向的数据越来越依赖伪 4D 标注**
   - Kinema4D 用 ST-V2 把 DROID/Bridge/RT-1 这类 2D 机器人视频 lift 成 pointmap。
   - RoboStereo 用 DepthAnything V3 从 Bridge V2 生成 XYZ pointmap。
   - Pri4R 在真实数据中用 off-the-shelf 3D tracker 生成 point track 伪标签。
   - 这说明 4D world model 的数据瓶颈正在通过 foundation geometry models 缓解，但伪标注质量仍是关键风险。

2. **视频生成方向强调“训练时暴露真实推理噪声”**
   - Vista4D 明确用 noisy multiview 4D reconstruction 训练，而不是只用干净 depth。
   - VerseCrafter 通过自动数据引擎构造 4D control maps，使真实视频也能提供几何控制监督。

3. **评测数据从视觉质量转向几何/动作有效性**
   - Embody4D、Vista4D、VerseCrafter 仍报告 VBench/FID/FVD/LPIPS 等视觉指标。
   - X-WAM、Pri4R、MVISTA、RoboStereo 更重视 manipulation success rate。
   - AW4RE 强调 evidence-backed 区域指标和 temporal consistency。

4. **DynamicVerse 类数据集可能成为通用 4D 模型的基础设施**
   - 它不仅提供 depth/camera，还提供 mask、object/category、object/scene/camera captions。
   - 这类多模态 4D 标注可服务未来 4D VLM、4D video generation 和 embodied world model。

---

## 14. 读后综合判断

这批论文显示 4D world model 正在从“能生成好看的动态视频”转向三个更具体的方向：

1. **几何可验证**
   - 仅 RGB 视频不够，越来越多工作输出 depth、XYZ pointmap、point tracks、4D Gaussian 或 point cloud。
   - 代表：X-WAM、Kinema4D、RoboStereo、MVISTA-4D、Vista4D。

2. **动作/控制可闭环**
   - 世界模型不只想象未来，还要用于 action decoding、test-time verification、policy optimization。
   - 代表：X-WAM、RoboStereo、Pri4R、MVISTA-4D。

3. **数据自动化**
   - 真实 4D 数据极缺，主流做法是用 VLM/GFM/segmenter/tracker/depth estimator 自动生成伪 4D 标注。
   - 代表：Embody4D、Kinema4D、VerseCrafter、DynamicVerse。

最大的开放问题仍然是：这些伪 4D 表征是否足够物理可靠，能否在接触、遮挡、非刚体、长期闭环 rollout 中不累积错误。现阶段最有实用价值的路线可能不是纯粹追求更大的视频生成模型，而是把 **显式几何约束、动作结构先验、数据自动化质量控制** 三者结合起来。

---

## 15. 面向 4D world model rollout 的训练/finetune 需求讨论

### 15.1 记录的用户问题

> 现在我想从 4D world model 切入，尝试利用 4DWM 的能力做 rollout。我认为，在 training-free 的设定下的工作都是小打小闹，去做真正有用的东西一定要上手进行训练。你看过这些文章，你觉得从需求、数据支持等方面而言，有什么地方或需求还是目前这些 4D WM 做不好或者没法解决的地方，必须要进行训练或 finetune 才能解决的呢？比如传统第三人称到 egoview 的 gap，第三人称到 embodiment 动作的 gap 等等。

### 15.2 总体判断

如果目标是用 4D world model 做真正可用于机器人决策的 rollout，最值得训练或 finetune 的不是“任意视角生成”本身，而是下面几类能力：

1. 动作条件：给定 candidate action，世界会如何变化。
2. embodiment 条件：同一个任务意图在不同机器人形态、相机安装、控制空间下如何展开。
3. 接触条件：夹爪、物体、桌面、遮挡和非刚体动态如何相互作用。
4. 策略接口：rollout 输出能否被 policy、reward model、verifier 直接使用。

training-free 方法最多能做几何重投影、视角补全、视频修复或弱伪标签生成。一旦进入闭环 rollout，它们缺少因果动作-世界动态，容易变成“好看的视频预测”，而不是“可用于决策的 mental simulator”。

### 15.3 第三人称视频到 egoview / wrist-view 的 gap

这是很现实的切入点。很多机器人数据，尤其 Bridge、DROID、RT-1、OpenX 体系，第三人称视角多；但策略执行时常依赖 wrist camera、head camera 或 ego view。

training-free novel view synthesis 只能解决一部分几何视角转换，解决不了：

- wrist camera 会被机械臂、夹爪、自身结构遮挡；
- egoview 中物体尺度、接触区域、可见边界和第三人称完全不同；
- 机器人执行动作时相机随 embodiment 一起运动，视角变化与 action/proprioception 强绑定；
- 第三人称中可见的物体，在 wrist view 可能不可见，反之亦然；
- 关键操作细节，例如夹爪是否对齐、接触点是否闭合，通常只在 wrist/ego view 更清楚。

因此这里需要训练一个 **view- and embodiment-conditioned 4D rollout model**。

理想输入：

- 第三人称初始图像或视频；
- 机器人 URDF 或 embodiment token；
- wrist/head camera extrinsics；
- proprioception；
- candidate action sequence。

理想输出：

- 未来 wrist/egoview RGB-D；
- 未来 pointmap 或 point tracks；
- gripper-object contact mask；
- uncertainty / visibility mask。

可借鉴的论文元素：

- Embody4D：单目到任意视角；
- Kinema4D：URDF + IK/FK 保证机器人轨迹精确；
- RoboStereo / X-WAM：RGB-D 4D rollout；
- Pri4R：3D point tracks 作为训练监督。

但现有论文还没有很好地把 **third-person demonstration -> policy egoview rollout** 做成一个强应用。

### 15.4 第三人称观察到 embodiment action 的 gap

第三人称视频只告诉模型“发生了什么”，不告诉模型“某个具体机器人该怎么动”。这个 gap 比视角 gap 更深，因为它涉及动作空间和 embodiment：

- 不同机器人 morphology 不同；
- 夹爪宽度、末端位姿、关节限制、速度限制不同；
- 第三人称视频中的运动可能来自人手或另一种机械臂，不能直接映射到目标机器人；
- 同样视觉变化可以由多种动作导致，inverse dynamics 本身病态；
- 真实控制器还有延迟、阻抗、夹爪闭合策略等低层差异。

training-free 方法基本不可能解决这个问题，因为它缺少动作-结果的统计对应关系。需要训练的是 **embodiment-aware inverse / forward 4D dynamics**。

forward rollout 形式：

- 输入：当前 4D scene state、目标机器人 embodiment、action chunk、camera setup。
- 输出：未来 RGB-D / pointmap、object displacement、contact/grasp state、success likelihood。

inverse action inference 形式：

- 输入：想象出的未来 4D trajectory、当前机器人 embodiment、task instruction。
- 输出：可执行 action chunk。

MVISTA-4D 的 trajectory latent + residual inverse dynamics 是一个有价值的方向。但进一步可以考虑训练跨 embodiment 的 action latent space：让不同机器人共享“任务级动作意图”，再由 embodiment decoder 转成具体控制。

### 15.5 接触、遮挡、失败动作和 near-miss rollout

很多 4D world model 能生成视觉上合理的结果，但机器人 rollout 真正需要判断：

- 是否发生接触；
- 是否抓稳；
- 是否推歪；
- 夹爪是否穿模；
- 物体是否滑落；
- near-miss 是失败还是仍有恢复空间；
- action 轻微偏差会不会造成完全不同结果。

这些属于 **contact-rich dynamics**，training-free 几何补全很难处理。Kinema4D 强调 near-miss 和失败执行，这是正确方向；但这类能力依赖数据中包含：

- 成功轨迹；
- 失败轨迹；
- 接触前后物体运动；
- gripper-object 相对位姿；
- force/tactile 信号，如果有会更好；
- object material、compliance、friction 等隐变量。

真正有用的 4D rollout 不应只输出视频，还应输出：

- contact probability；
- grasp stability；
- object pose / point tracks；
- collision / penetration risk；
- task success score；
- uncertainty。

基础视频模型有视觉先验，但没有足够的机器人接触先验，因此这个方向必须通过机器人交互数据 finetune。

### 15.6 长时域 closed-loop rollout 的误差累积

很多论文生成 49 帧或 81 帧，看起来已经较长，但机器人策略 rollout 是闭环过程：

1. policy 输出 action；
2. world model rollout；
3. policy 根据 imagined observation 再输出 action；
4. 多轮循环。

这里会出现严重误差累积：

- 几何漂移；
- 物体尺度漂移；
- 接触状态漂移；
- 机器人自身状态和视觉状态不一致；
- 小错误在几轮后变成完全错误的世界。

training-free 方法无法学习这种 closed-loop distribution。必须用包含多步交互的数据训练，尤其是：

- action-conditioned chunks；
- autoregressive rollout；
- re-observation correction；
- uncertainty-aware belief update。

AW4RE 的 evidence-backed / unsupported 区域分离很有启发：机器人 rollout 也应该输出“哪些区域是模型有证据的，哪些是幻觉补全的”。否则策略会过度相信 hallucinated future。

### 15.7 几何表征和策略接口之间的 gap

很多 4D world model 输出 RGB video 或 RGB-D video，但策略真正需要的可能不是视频，而是：

- 目标物体 6D pose；
- affordance；
- grasp candidates；
- object point tracks；
- collision-free regions；
- success/failure score；
- subgoal state；
- uncertainty map。

如果只训练视频生成，policy 还要再从视频中解析这些信息，链路很长、误差很大。

所以值得训练的不是单纯 video model，而是 **rollout + task-relevant head**：

- RGB-D head；
- pointmap head；
- object track head；
- contact head；
- success head；
- reward head；
- action-consistency head。

Pri4R 的启发是：3D point tracks 作为 privileged supervision，可能比直接生成 RGB 更能帮助策略学习。对 rollout 来说，也可以用 point tracks 或 object-centric 4D state 作为核心监督，而不是只追求视频质量。

### 15.8 数据支持方面的主要缺口

#### 15.8.1 Paired third-person + egoview/wrist-view + action 数据

这类数据用于训练第三人称到策略视角的 4D 转换。理想数据应同步包含：

- third-person RGB-D；
- wrist/ego RGB-D；
- robot proprioception；
- action chunk；
- camera extrinsics；
- task language；
- success/failure label。

这类数据直接对应 third-person to egoview gap。

#### 15.8.2 Cross-embodiment action-conditioned 4D 数据

这类数据用于解决第三人称观察到 embodiment action 的 gap。理想数据应包含：

- 多种机器人；
- URDF / kinematic chain；
- 同类任务在不同机器人上的执行；
- end-effector trajectory；
- joint action；
- object 4D trajectory。

这类数据可以训练 embodiment-conditioned rollout model，让模型知道同一个 task intention 在不同 embodiment 下如何导致不同视觉/几何结果。

#### 15.8.3 Failure-rich manipulation 数据

现有 imitation 数据成功轨迹偏多，但 rollout 用于决策时，最重要的是识别坏 action。需要收集：

- failed grasps；
- near-miss；
- collision；
- slipping；
- occluded failure；
- recovery attempts；
- human/robot interventions。

没有这类数据，world model 很容易 optimistic hallucination：把不合理 action 也生成成成功结果。

#### 15.8.4 Contact / force / tactile 对齐数据

纯视觉 4D 对接触理解不够。如果能有：

- tactile image；
- force torque；
- gripper current；
- contact event labels；
- object pose tracking；

就可以训练更可靠的 contact-aware rollout。

### 15.9 最有价值的研究切入点

#### 方向一：Third-person-to-egoview 4D rollout for policy learning

目标：把大量第三人称机器人数据转成策略可用的 ego/wrist rollout 数据。

模型：

- 输入 third-person observation + action/proprio + camera/URDF；
- 输出 future wrist/egoview RGB-D + pointmap + contact mask。

价值：

- 直接解决数据视角不匹配；
- 能把已有大规模第三人称数据变成 VLA/policy 可训练数据；
- 比单纯 novel view generation 更贴近机器人需求。

#### 方向二：Embodiment-conditioned action rollout

目标：给定机器人 embodiment 和 candidate action，预测未来 4D 世界。

模型：

- 输入 initial RGB-D/pointmap + URDF + proprio + action chunk；
- 输出 future RGB-D/pointmap/object tracks/success score。

价值：

- 可用于 test-time action verification；
- 可用于 model-based RL；
- 可用于 synthetic data generation；
- 可解决“第三人称观察到具体机器人动作”的 gap。

#### 方向三：Failure-aware 4D mental simulator

目标：不只是生成成功未来，而是判断动作会不会失败。

模型输出：

- future 4D state；
- success probability；
- contact / collision / slip；
- uncertainty；
- recoverability。

价值：

- 对真实机器人部署最有用；
- 比生成漂亮视频更容易体现 4D world model 的决策价值；
- 当前论文普遍做得不够。

### 15.10 简洁结论

training-free 4D 方法适合做：

- 视角补全；
- 几何重投影；
- video reshooting；
- 数据预处理；
- weak pseudo-label。

但如果目标是机器人 rollout，必须训练/finetune 的能力是：

1. 动作导致世界如何变化；
2. 某个 embodiment 如何执行这个动作；
3. 第三人称视觉如何变成策略真实可见的 egoview/wrist-view；
4. 接触、失败、遮挡和恢复如何发生；
5. rollout 输出如何服务策略，而不只是生成视频。

可以把一个有潜力的研究题目表述为：

> 训练一个 embodiment-aware、view-transferable、failure-aware 的 4D rollout model，把第三人称机器人数据转化为目标机器人 egoview/wrist-view 下的动作条件未来，并输出策略可用的几何、接触和成功信号。
