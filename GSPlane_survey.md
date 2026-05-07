# GSPlane 相关工作调研：2025 年以来的平面重建、平面先验与 Gaussian Splatting

> 论文主题：**GSPlane: Concise and Accurate Planar Reconstruction via Structured Representation**  
> 调研日期：2026-05-07  
> 检索关键词示例：`Gaussian Splatting Planar`、`Planar Prior Reconstruction`、`Gaussian Splatting Prior`、`Planar Gaussian Splatting`、`plane prior 3DGS`、`Manhattan/Atlanta world Gaussian Splatting`、`depth normal prior Gaussian Splatting meshing`、`planar splatting reconstruction`。

## 1. 纳入标准与阅读口径

本文件主要收录自 2025 年初以来公开发表或公开预印本中，与 GSPlane 题目存在实质相关性的工作：

1. **直接相关**：以平面、平面先验、平面 primitive、planar splatting、3D plane parsing/reconstruction 为核心的 3DGS 或 planar representation 方法。
2. **强相关**：虽然不直接输出 plane instance，但使用深度、法线、语义、Manhattan/Atlanta-world 等结构知识解决 3DGS 在低纹理平面区域的几何退化。
3. **可作为 GSPlane 知识来源或评测参照**：提供平面检测/单图平面重建、foundation-model 几何先验、平面/结构约束、mesh/TSDF/三角 primitive 与 Gaussian 的耦合思路。

注意：PGSR 是 2024 年工作，但在 2025 后多篇论文中作为高频基线和方法基础出现，因此单独列为“背景基线”，不计入 2025 后主时间线。

## 2. 按时间顺序的论文总览

| 时间 | 论文 | 相关性 | 是否开源 | 核心知识/先验如何进入 workflow | 额外 off-the-shelf 模型/工具 | 评测/实验数据集 |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 WACV | [Planar Gaussian Splatting](https://arxiv.org/abs/2412.01931) | 直接相关：3DGS 中解析 3D 平面实例 | 未找到官方代码 | 将 SAM 2D mask、法线和可学习 plane descriptor lift 到 3D Gaussian；用层级 Gaussian mixture tree 合并相似 Gaussian 得到 plane instance | SAM；Omnidata normal | ScanNetV2、Replica |
| 2025 WACV | [DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing](https://arxiv.org/abs/2403.17822) | 强相关：深度/法线先验改善室内低纹理平面几何 | [maturk/dn-splatter](https://github.com/maturk/dn-splatter) | 将传感器/单目深度作为 depth loss，将 Gaussian 最短轴作为 normal 并用 monocular normal 监督；再用 Poisson/TSDF 类流程抽 mesh | ZoeDepth、DepthAnything、Omnidata；可用 RGB-D sensor depth | MuSHRoom、ScanNet++ |
| 2025-05 | [OmniIndoor3D: Comprehensive Indoor 3D Reconstruction](https://arxiv.org/abs/2505.20610) | 相邻相关：panoptic/geometry 共同优化，含平面平滑 densification | 项目页标注 code coming soon；GitHub 仓库存在但公开内容有限：[ucwxb/OmniIndoor3D](https://github.com/ucwxb/OmniIndoor3D) | RGB-D 粗重建初始化 Gaussian；轻量 MLP 解耦外观/几何；panoptic-guided densification 鼓励 planar surface smoothness | Grounded SAM 生成 pseudo semantic/instance；RGB-D sensor | ScanNet、ScanNet++ |
| 2025 CVPR | [IndoorGS: Geometric Cues Guided Gaussian Splatting for Indoor Scene Reconstruction](https://openaccess.thecvf.com/content/CVPR2025/html/Ruan_IndoorGS_Geometric_Cues_Guided_Gaussian_Splatting_for_Indoor_Scene_Reconstruction_CVPR_2025_paper.html) | 强相关：室内线/点/面几何 cue 引导 3DGS | 未找到官方代码；有非官方复现 [NJUCG/IndoorGS-exp](https://github.com/NJUCG/IndoorGS-exp) | 从 2D line、SfM 点、3D plane cue 构建几何引导；用于初始化和 geometric-cue-guided adaptive density control | 2D line extractor/feature matching；复现中使用 COLMAP、LIMAP、SAM | 公开摘要称多个室内数据集，具体集合需查全文 |
| 2025 CVPR | [PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes](https://arxiv.org/abs/2412.03451) | 直接相关：直接优化 3D plane primitive | [ant-research/PlanarSplatting](https://github.com/ant-research/PlanarSplatting) | 以矩形 3D plane primitive 为几何主体；通过 CUDA planar splatting 渲染 2.5D depth/normal 并优化；后处理 merge plane primitive | Metric3Dv2 用于粗深度/初始化；现代 monocular depth/normal cues | ScanNetV2、ScanNet++ |
| 2025 CVPR | [ZeroPlane: Towards In-the-wild 3D Plane Reconstruction from a Single Image](https://arxiv.org/abs/2506.02493) | 可作为外部平面知识来源：单图零样本 3D plane detector/reconstructor | [jcliu0428/ZeroPlane](https://github.com/jcliu0428/ZeroPlane) | Transformer query 预测 plane mask、normal、offset；可替代 SAM+normal heuristic，作为 GSPlane/PlanarGS 的 2D plane prior source | DINOv2、DPT/RefineNet；Mask2Former 用于部分数据标签生成 | 训练/验证含 ScanNetV1/V2、Matterport3D、Replica、HM3D、DIODE、Taskonomy、Synthia、Virtual KITTI、Sanpo 等；zero-shot 评测含 NYUv2、7-Scenes、ApolloScape、ParallelDomain |
| 2025-07 | [SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](https://arxiv.org/abs/2507.15602) | 相邻相关：SDF 与 3DGS 双向增强，解决全局几何一致性 | [aim-uofa/SurfaceSplat](https://github.com/aim-uofa/SurfaceSplat) | SDF 先产生粗 mesh 供 3DGS 初始化；3DGS 再生成 novel views 反哺 SDF，形成 surface reconstruction 与 rendering 的闭环 | 主要是 SDF/3DGS 组合，未突出外部 foundation model | DTU、MobileBrick |
| 2025 IEEE Access | [AnchorGS: Anchoring Gaussian Splatting to Planar Priors for Robust 3D Reconstruction](https://doaj.org/article/af420b9f22eb4f25aac37a96f707d621) | 直接相关：显式 plane prior anchor 3DGS | 未找到官方代码 | 多阶段层级优化：plane attraction、normal alignment、thinness constraint；专门 densification 填补 planar holes | 显式几何 plane prior，公开摘要未说明具体检测模型 | 公开摘要未列出具体数据集 |
| 2025 KSII TIIS | [Leveraging Planar Prior Knowledge for Regularization in 3D Gaussian Splatting Optimization](https://koreascience.kr/article/JAKO202523836005559.do) | 直接相关：plane equation regularization for 3DGS | 未找到官方代码 | 用 predefined/RANSAC plane equation 约束 Gaussian mean 到平面距离；动态更新 plane equation；dot loss 约束 plane normal 一致性 | 公开摘要指向 plane prior/RANSAC，未发现额外模型说明 | 公开页面获取受限，需查全文确认 |
| 2025-10 | [G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior](https://arxiv.org/abs/2510.12099) | 强相关：用平面结构获得 metric depth，再引导 generative prior | [DaLi-Jack/G4Splat](https://github.com/DaLi-Jack/G4Splat) | 提取全局 3D planes，生成 plane-aware depth；用于 observed/unobserved regions 的几何监督、visibility mask、novel view selection、video diffusion inpainting consistency | SAM、monocular/depth-normal predictor、MASt3R-SfM/MAtCha、video diffusion model | Replica、ScanNet++、DeepBlending、Mip-NeRF 360 |
| 2025-10 | [GSPlane: Concise and Accurate Planar Reconstruction via Structured Representation](https://arxiv.org/abs/2510.17095) | 目标论文 | 未找到官方代码 | SAM+normal 生成 2D planar masks；投影到 3D point/Gaussian graph；Leiden 聚类得到 plane groups；将 planar Gaussian xyz 重参数化为三个 basis points 的加权组合；DGR 动态剔除误分平面 Gaussian；mesh layout refinement 与 supportive plane correction | SAM、Metric3Dv2、COLMAP/SfM、RANSAC、Leiden graph clustering；可叠加 3DGS/2DGS/GOF/RaDe-GS/PGSR | ScanNetV2、Tanks and Temples |
| 2025-10 | [PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](https://arxiv.org/abs/2510.18714) | 直接相关：feed-forward planar splatting，平面 primitive 紧凑表示 | [lck666666/plana3r](https://github.com/lck666666/plana3r) | ViT 从双视图直接预测 sparse planar primitives 与相对 pose；用 PlanarSplatting renderer 将 primitives 渲染为 depth/normal 作监督；无需 plane annotation | DUSt3R pretrained weights；Metric3Dv2 生成 pseudo normal；PlanarSplatting CUDA renderer | 训练：ScanNetV2、ScanNet++、ARKitScenes、Habitat；评测：ScanNetV2、Matterport3D、NYUv2、Replica，补充 7-Scenes |
| 2025 NeurIPS | [PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](https://arxiv.org/abs/2510.23930) | 直接相关：vision-language planar priors 引导 3DGS | [SJTU-ViSYS-team/PlanarGS](https://github.com/SJTU-ViSYS-team/PlanarGS) | LP3 pipeline 用 language prompt 得到 planar region proposals，经 cross-view fusion 和 geometric inspection 修正；训练中加入 plane-guided initialization、Gaussian flattening、co-planarity constraint、depth/normal geometric prior | GroundedSAM/Grounding DINO+SAM；DUSt3R depth/normal | Replica、ScanNet++、MuSHRoom |
| 2025 NeurIPS | [AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians](https://arxiv.org/abs/2510.25129) | 强相关：Atlanta-world/global structural plane regularization | [xyzhang77/AtlasGS](https://github.com/xyzhang77/AtlasGS) | 语义 Gaussian 预测 wall/floor/ceiling/other；显式 plane indicators 表示 floor/ceiling/vertical walls；3D global planar regularization + 2D local surface regularization 约束位置和法线 | 预训练 semantic segmentation；monocular geometry priors/depth-normal models | Replica、ScanNet++、ScanNet；另含 indoor/outdoor qualitative |
| 2026-01 | [PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction](https://arxiv.org/abs/2601.22046) | 相邻相关：triangle geometry + Gaussian appearance，显式 planar abstraction | [InternRobotics/PLANING](https://github.com/InternRobotics/PLANING) | 三角 primitive 负责显式几何，neural Gaussians 负责外观；streaming mapper 中分离 geometry/appearance 更新；重建 triangle soup 后可做 plane extraction | MASt3R/前馈点图模型提供 depth/normal/pose prior | ScanNet++、ScanNetV2、FAST-LIVO2 |
| 2026-03 | [3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](https://arxiv.org/abs/2603.19682) | 相邻相关：自生成 TSDF prior 约束 Gaussian 到 surface band | 项目页存在：[GSPrior](https://takeshie.github.io/GSPrior/)；代码状态未明确 | 周期性融合当前 rendered depth 为 TSDF grid；用 narrow band 删除 outlier Gaussians、约束 opacity、pull Gaussians toward surface；不依赖外部数据先验 | 无外部 foundation model；prior 从当前 Gaussian depth 自生成 | NeRF-Synthetic、DTU、Tanks and Temples、Mip-NeRF 360 |
| 2026-04 | [Shape-Optimized Gaussian Splatting for UAV Reconstruction with Manhattan Constraints](https://www.mdpi.com/2079-9292/15/8/1647) | 相邻相关：Manhattan structural prior 约束 UAV/urban GS | 未发现官方代码 | DeepLabV3+ 语义分组；Dynamic Sobel 提取建筑边缘；Shape Attention + Manhattan World 约束 Gaussian mean/covariance 与建筑主方向 | DeepLabV3+；Dynamic Sobel layer | GauUSceneV2、JN-Aerial、MatrixCity-Aerial |

## 3. 重点论文技术路线整理

### 3.1 GSPlane

- **问题定位**：常规 3DGS/2DGS/PGSR 等方法可以提升渲染或局部几何，但 planar regions 仍容易出现波浪、过密 mesh、拓扑不干净的问题。
- **知识来源**：
  - SAM 产生 subpart masks。
  - Metric3Dv2 预测 normal map。
  - SfM/COLMAP 初始点云提供 2D-to-3D 投影和 Gaussian 初始化。
- **workflow 中如何使用知识**：
  1. 对每个输入视角，以 SAM mask 为候选区域，利用 Metric3Dv2 normal 的区域一致性筛选 2D planar masks。
  2. 将 planar masks 投影到 3D point cloud，构造点之间“同属一个 planar mask”的加权图。
  3. 用 Leiden algorithm 聚类，得到 3D plane group。
  4. 对每个 plane group 用 RANSAC 拟合 plane，并将对应 Gaussian center 的 `xyz` 改写为三个非共线 basis points 的归一化加权组合。
  5. 训练中同时优化 basis points 与每个 Gaussian 的 weights，让 Gaussian 被严格限制在 plane 上。
  6. Dynamic Gaussian Re-classifier 监控高梯度 planar Gaussians，将疑似误分的 planar Gaussian 重新变回自由 `xyz` 表示。
  7. Mesh layout refinement 将平面区域投影、网格化、Delaunay 三角化，减少冗余顶点/面并改善拓扑；Supportive Plane Correction 支持桌面/地面等支撑平面的封洞与对象解耦。
- **对论文写作的启发**：
  - GSPlane 相比 PlanarGS 的差异不只是“多了平面监督”，而是将 planar prior 变成 **Gaussian coordinate parameterization**，再进一步影响 mesh topology。
  - 可突出“structured representation”对 downstream editability/simulation 的价值，而不仅是 Chamfer/F-score。

### 3.2 PlanarGS

- **路线**：在 3DGS 中引入 vision-language planar priors，面向 indoor low-texture surfaces。
- **知识 workflow**：
  1. 用 text prompts（如 wall、floor、door）驱动 GroundedSAM 得到候选平面区域。
  2. 用 DUSt3R 生成多视图一致 depth/normal。
  3. 对候选 mask 做 cross-view fusion，补全单视角漏检。
  4. 通过 normal clustering 和 plane-distance map 做 geometric inspection，拆分误合并平面、过滤非平面。
  5. 训练时加入 plane-guided initialization、Gaussian flattening、co-planarity constraint；同时用 DUSt3R depth/normal 做 geometric prior supervision。
- **外部模型**：GroundedSAM、DUSt3R。
- **与 GSPlane 的关系**：
  - PlanarGS 的 planar prior 主要作为监督项和初始化/flattening 规则；
  - GSPlane 将 Gaussian 坐标本身结构化到 plane basis 上，且有 DGR 和 mesh layout refinement。

### 3.3 PlanarSplatting

- **路线**：不从 Gaussian 出发，而是直接优化 3D rectangular plane primitives；通过 differentiable planar splatting 渲染 depth/normal。
- **知识 workflow**：
  1. 使用 Metric3Dv2 等 monocular geometry foundation model 生成粗深度/法线。
  2. 初始化少量 3D planar primitives。
  3. 用 CUDA plane splatting 将 plane primitives 渲染成 2.5D depth/normal map。
  4. 在 depth/normal supervision 下优化 plane center、rotation、radii。
  5. 合并相似 plane primitives 得到最终 planar reconstruction。
- **对 GSPlane 的价值**：
  - 证明显式平面 primitive 对结构化室内场景的效率优势；
  - GSPlane 则保持 3DGS pipeline 和渲染质量，同时在 planar subsets 上施加结构化约束。

### 3.4 Planar Gaussian Splatting (PGS)

- **路线**：在 Gaussian primitive 上引入 normal 和 plane descriptor，用 hierarchical Gaussian mixture tree 做 3D plane instance parsing。
- **知识 workflow**：
  1. Omnidata 预测 normal，监督 Gaussian normal rendering。
  2. SAM 产生 2D masks，经过 RAG 合并得到更接近平面的 2D segments。
  3. 将 2D segment label 通过可学习 descriptor lift 到 3D Gaussian。
  4. 用 recurrent mean-shift/holistic separability 让 descriptor 更可分。
  5. 层级 GMM 根据几何距离和 descriptor 相似度合并 Gaussian，形成 3D plane instances。
- **对 GSPlane 的价值**：
  - 与 GSPlane 同样从 2D segmentation + normal 中抽取平面知识；
  - 但 PGS 更偏 plane parsing/instance grouping，GSPlane 更偏 planar surface geometry + mesh topology。

### 3.5 IndoorGS

- **路线**：从室内常见几何 cue（线、点、面）入手，不依赖单一平面检测器。
- **知识 workflow**：
  1. 2D lines 经 feature matching 融合成 3D line cues。
  2. SfM points 经 statistical outlier removal 清洗。
  3. 在 textureless regions 提取 3D plane cues。
  4. 各类 cue 被用于初始化和 adaptive density control，使 Gaussian 增密更符合室内结构。
- **对 GSPlane 的价值**：
  - 可作为“非 foundation-model 平面 cue extraction”的对照；
  - 适合讨论 line/plane/SfM multi-cue 与 SAM/Metric3Dv2 的差别。

### 3.6 ZeroPlane

- **路线**：大规模跨域单图 3D plane reconstruction，可作为 GSPlane/PlanarGS 的 off-the-shelf plane detector 候选。
- **知识 workflow**：
  1. DINOv2 + DPT-style decoder 提取多尺度图像特征。
  2. Transformer plane queries 同时预测 plane mask、plane normal、plane offset。
  3. normal 与 offset 解耦，并使用 classification-then-regression 提升跨域泛化。
  4. 构建 14+ datasets、56 万高分辨率 dense planar annotations。
- **对 GSPlane 的价值**：
  - GSPlane 当前用 SAM mask + normal consistency heuristic；ZeroPlane 提供更直接的 plane instance/mask/parameter prior。
  - 可在论文讨论中作为 future work 或替代 prior source。

### 3.7 AtlasGS

- **路线**：用 Atlanta-world assumption 对 wall/floor/ceiling 等低纹理结构进行全局正则。
- **知识 workflow**：
  1. 预训练语义分割模型生成 wall/floor/ceiling pseudo labels。
  2. 语义 Gaussian 表示每个 Gaussian 属于结构区域的概率。
  3. 显式 plane indicators 表示地板/天花板/墙等全局结构平面。
  4. 3D global planar regularization 对 Gaussian position/normal 施加结构约束。
  5. 2D local surface regularization 从 rendered depth 推导 surface normal 并对齐结构方向。
- **对 GSPlane 的价值**：
  - AtlasGS 的全局结构假设强，适合 Manhattan/Atlanta scenes；
  - GSPlane 的 plane basis 表示更通用，可覆盖桌面、柜面、街道立面等非固定语义的 planar regions。

### 3.8 PLANA3R

- **路线**：从 unposed two-view images 直接 feed-forward 预测 planar primitives 和 relative pose。
- **知识 workflow**：
  1. ViT/DUSt3R 风格编码双视图。
  2. hierarchical primitive prediction architecture 预测不同分辨率的 planar primitives。
  3. 通过 PlanarSplatting renderer 渲染高分辨率 depth/normal，使用 depth/normal supervision 训练。
  4. 不需要显式 plane annotation，训练规模可依托 RGB-D/stereo datasets。
- **对 GSPlane 的价值**：
  - 提供“无需 per-scene optimization 的 planar primitive prior”方向；
  - GSPlane 面向 posed multi-view optimization，PLANA3R 面向 feed-forward pose-free 设定。

### 3.9 G4Splat

- **路线**：针对 sparse-view/unobserved region，先用 global planes 取得 metric-scale depth，再把几何知识注入 generative inpainting loop。
- **知识 workflow**：
  1. SAM+normal/depth 产生 per-view plane masks。
  2. 利用 scene point cloud 合并成 global 3D planes。
  3. 对 planar regions 用 ray-plane intersection 生成 plane-aware depth。
  4. plane-aware depth 建 visibility grid，指导 novel view selection 和 inpainting mask。
  5. 视频扩散模型生成未观测区域，多视角监督时用 plane grouping 降低跨视图冲突。
- **对 GSPlane 的价值**：
  - 强调 plane 不只是重建目标，也可以成为 sparse-view completion/generative prior 的几何锚点。

### 3.10 PLANING

- **路线**：用 triangle primitive 作为显式几何，neural Gaussian 作为外观，适配 streaming reconstruction。
- **知识 workflow**：
  1. 前端/后端估计相机位姿和 dense point maps。
  2. mapper 初始化 learnable triangles，使用 depth/normal priors 监督几何。
  3. 每个 triangle anchor 若干 neural Gaussians 表示外观。
  4. reconstruction 后从 triangle soup 中做 coarse-to-fine plane extraction。
- **对 GSPlane 的价值**：
  - 展示“显式几何 primitive + Gaussian appearance”在 simulation-ready 场景中的价值；
  - GSPlane 可在讨论中强调 mesh layout refinement 也朝向 simulation/editability。

## 4. 评测数据集统计

下表统计本次调研条目中明确出现的数据集。由于部分期刊/会议页面仅公开摘要，若全文中有更多数据集而公开摘要未列出，则未强行计入。

| 数据集 | 出现频次（约） | 使用论文/方法 | 主要用途 |
| --- | ---: | --- | --- |
| ScanNet / ScanNetV2 | 8 | PGS、PlanarSplatting、ZeroPlane、OmniIndoor3D、GSPlane、PLANA3R、AtlasGS、PLANING | 室内 planar reconstruction、surface reconstruction、training/evaluation |
| ScanNet++ | 8 | PlanarSplatting、DN-Splatter、OmniIndoor3D、G4Splat、PLANA3R、PlanarGS、AtlasGS、PLANING | 高保真室内几何/mesh/NVS |
| Replica | 6 | PGS、ZeroPlane、G4Splat、PLANA3R、PlanarGS、AtlasGS | synthetic indoor，几何/渲染/平面评测 |
| MuSHRoom | 2 | DN-Splatter、PlanarGS | real-world indoor mesh/NVS |
| Tanks and Temples | 3 | PGSR（背景）、GSPlane、GSPrior | outdoor/large-scale surface reconstruction |
| DTU | 3 | PGSR（背景）、SurfaceSplat、GSPrior | object/scene surface reconstruction |
| Mip-NeRF 360 | 4 | PGSR（背景）、ZeroPlane、G4Splat、GSPrior | outdoor/indoor NVS 与 sparse-view/generalization |
| Matterport3D | 2 | ZeroPlane、PLANA3R | 平面标签训练/跨数据评测 |
| NYUv2 | 2 | ZeroPlane、PLANA3R | zero-shot/indoor evaluation |
| 7-Scenes | 2 | ZeroPlane、PLANA3R | zero-shot/indoor evaluation |
| ARKitScenes | 2 | ZeroPlane、PLANA3R | RGB-D indoor training/generalization |
| Habitat / HM3D | 2 | ZeroPlane、PLANA3R | synthetic indoor training/generalization |
| DeepBlending | 1 | G4Splat | sparse-view reconstruction/rendering |
| MobileBrick | 1 | SurfaceSplat | sparse-view surface reconstruction |
| NeRF-Synthetic | 1 | GSPrior | synthetic geometry/rendering |
| FAST-LIVO2 | 1 | PLANING | streaming/SLAM-style reconstruction |
| GauUSceneV2 | 1 | Shape-Optimized GS/MS-GS | UAV/urban reconstruction |
| JN-Aerial | 1 | Shape-Optimized GS/MS-GS | UAV/urban reconstruction，新建 Jinan dataset |
| MatrixCity-Aerial | 1 | Shape-Optimized GS/MS-GS | aerial/urban reconstruction |
| ApolloScape | 1 | ZeroPlane | outdoor zero-shot plane evaluation |
| ParallelDomain | 1 | ZeroPlane | outdoor zero-shot plane evaluation |

### 数据集使用趋势

1. **室内结构化重建主战场**：ScanNet/ScanNetV2、ScanNet++、Replica、MuSHRoom 是 PlanarGS/GSPlane/PlanarSplatting/DN-Splatter 等工作的共同评测核心。
2. **平面重建泛化**：ZeroPlane 和 PLANA3R 开始大量引入 Matterport3D、NYUv2、7-Scenes、ARKitScenes、Habitat/HM3D 等跨域数据，目标从 per-scene optimization 扩展到 feed-forward/zero-shot。
3. **室外/城市平面结构**：GSPlane 使用 Tanks and Temples 验证室外平面；AtlasGS/G4Splat/MS-GS 将 Atlanta/Manhattan/plane-aware depth 扩展到 urban 或 aerial scenes。
4. **稀疏视角与 generative prior**：G4Splat、SurfaceSplat、GSPrior 更强调 sparse-view 或 global geometry coherence，虽然不是纯 planar reconstruction，但与 GSPlane 的“外部知识如何稳定几何”问题高度相关。

## 5. 对 GSPlane 论文定位的建议

1. **与 PlanarSplatting 区分**：PlanarSplatting 直接以 3D plane primitive 为主体，速度快、结构紧凑；GSPlane 保留 Gaussian Splatting 主体，在已检测 plane regions 上重参数化 Gaussian 坐标，并进一步改善 mesh topology。
2. **与 PlanarGS 区分**：PlanarGS 的核心是 vision-language planar priors + co-planarity/depth/normal supervision；GSPlane 的核心是 structured planar coordinate representation + DGR + mesh layout refinement。
3. **与 PGS 区分**：PGS 侧重无监督 plane instance parsing；GSPlane 侧重平面区域的几何精度、拓扑干净度和编辑/仿真可用性。
4. **与 AtlasGS/Manhattan 类方法区分**：Atlanta/Manhattan-world 假设适合 wall/floor/ceiling/urban facade，但语义和方向假设较强；GSPlane 的 plane cluster 来自图聚类和 local planar priors，语义限制更弱。
5. **可强调 workflow knowledge 的层级**：
   - 低层几何知识：normal/depth。
   - 中层区域知识：SAM/ZeroPlane/GroundedSAM masks。
   - 高层结构知识：plane groups、basis points、mesh topology。
   - 训练鲁棒性知识：DGR 用梯度统计识别 false-positive planar Gaussians。
6. **未来扩展方向**：
   - 用 ZeroPlane 或 PLANA3R 作为更强的 2D/3D planar prior source，替换或补充 SAM+normal heuristic。
   - 结合 PlanarGS 的 language prompts，为特定 supportive planes（table、floor、shelf）生成可控编辑标签。
   - 结合 G4Splat 的 plane-aware depth，在 sparse-view 或 unobserved regions 中让 GSPlane 支持补全。
   - 结合 PLANING/triangle primitive，将 GSPlane 的 refined planar mesh 直接导出为 simulation-ready assets。

## 6. 参考链接索引

- GSPlane: <https://arxiv.org/abs/2510.17095>
- PlanarGS: <https://arxiv.org/abs/2510.23930>, <https://github.com/SJTU-ViSYS-team/PlanarGS>
- PlanarSplatting: <https://arxiv.org/abs/2412.03451>, <https://github.com/ant-research/PlanarSplatting>
- Planar Gaussian Splatting: <https://arxiv.org/abs/2412.01931>
- DN-Splatter: <https://arxiv.org/abs/2403.17822>, <https://github.com/maturk/dn-splatter>
- IndoorGS: <https://openaccess.thecvf.com/content/CVPR2025/html/Ruan_IndoorGS_Geometric_Cues_Guided_Gaussian_Splatting_for_Indoor_Scene_Reconstruction_CVPR_2025_paper.html>
- ZeroPlane: <https://arxiv.org/abs/2506.02493>, <https://github.com/jcliu0428/ZeroPlane>
- OmniIndoor3D: <https://arxiv.org/abs/2505.20610>, <https://ucwxb.github.io/OmniIndoor3D/>
- SurfaceSplat: <https://arxiv.org/abs/2507.15602>, <https://github.com/aim-uofa/SurfaceSplat>
- AnchorGS: <https://doaj.org/article/af420b9f22eb4f25aac37a96f707d621>
- G4Splat: <https://arxiv.org/abs/2510.12099>, <https://github.com/DaLi-Jack/G4Splat>
- PLANA3R: <https://arxiv.org/abs/2510.18714>, <https://github.com/lck666666/plana3r>
- AtlasGS: <https://arxiv.org/abs/2510.25129>, <https://github.com/xyzhang77/AtlasGS>
- PLANING: <https://arxiv.org/abs/2601.22046>, <https://github.com/InternRobotics/PLANING>
- GSPrior / Self-Constrained Priors: <https://arxiv.org/abs/2603.19682>, <https://takeshie.github.io/GSPrior/>
- Shape-Optimized GS with Manhattan Constraints: <https://www.mdpi.com/2079-9292/15/8/1647>
- PGSR（背景基线）: <https://arxiv.org/abs/2406.06521>, <https://github.com/zju3dv/pgsr>
