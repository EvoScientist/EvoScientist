# SIGIR 2026综述-EvoScientist
## 摘要

**研究范围：** 本文基于截至 2026-04-16 已公开在 arXiv 的 68 篇论文，对 SIGIR 2026 full paper 的研究图景进行系统性综述。**核心观察：** 与传统把 SIGIR 理解为“检索模型改进会议”的视角不同，这批论文呈现出一个更分层的信息访问栈：推荐系统正在从表示学习与匹配函数优化，转向生成式推荐、项目 tokenization、偏好对齐与可验证推理[4,6,27,28]；检索研究则从“提高排序分数”扩展为多模态证据基础设施，重点落在 token 级效率、局部证据定位、混合模态检索与面向下游生成的检索接口[29,31,39,44]；Search Agent 与 RAG 方向进一步把搜索重写为轨迹级策略学习问题，其中 query refinement、trajectory supervision、不确定性估计与推理路由成为关键模块[46,54,56,60]；与此同时，评测、公平与对齐不再是外围议题，而开始直接进入系统目标函数与评估协议设计[61,63,64,65]。**代表性信号：** DIGER [6] 将 semantic ID 可微化，直接触及生成式推荐中的离散符号瓶颈；GEMS [26] 则尝试在统一 LLM 中联通搜索与推荐，显示任务边界本身正在被重写。**全文组织：** 本文首先给出主题图谱与统一任务视角，随后分别综述推荐与广告、检索与多模态访问、Search Agents/RAG/推理、评测公平与对齐四条主线，并补充评测维度与开放问题分析。**总体判断：** SIGIR 2026 最强的信号并不是单一模型家族的胜出，而是“evidence-grounded, controllable, agentic information access”正在成为新的共同范式[6,26,28,44,46,59]。

## 1. 引言

**问题背景：** 从这 68 篇论文可以看到，SIGIR 2026 的最大变化不是某个 backbone 再次刷新榜单，而是信息访问系统的目标函数和系统边界同时被改写了。推荐论文不再满足于在固定 item ID 空间里做排序，而是开始显式讨论生成、tokenization、偏好推理、测试时控制与用户满意度对齐[4,7,25,27,28]。其中，DIGER [6] 尤其值得注意，因为它把 semantic ID 从离散映射改写为可微对象，使 item identity 与生成目标之间第一次形成更直接的端到端优化接口。检索论文也不再只围绕相关性打分器本身展开，而是把更多精力放在“证据如何被选中、压缩、定位、重组并喂给下游模型”这一更宽的接口问题上[29,31,34,39,44]。Search/RAG 方向则把单轮查询处理推进到多步轨迹建模，开始把检索看成策略执行链条中的一步，而不是系统的终点[46,48,54,56]；而 GEMS [26] 则进一步表明，搜索与推荐本身也可能只是同一用户建模问题在不同子空间中的两种投影。

**核心判断：** 这批论文共同显示出三个结构性变化：（1）**推荐正在生成化与语义接口化**，问题重心从匹配函数转向 tokenization、推理与对齐[4,6,27,28]；（2）**检索正在基础设施化**，其价值越来越体现在证据组织、局部 grounding 和下游生成支持上[29,31,39,44]；（3）**搜索正在轨迹化与 agent 化**，系统需要学习何时检索、如何检索以及何时继续推理[46,54,56,60]。与此同时，评测、公平与治理不再是外围约束，而开始直接影响系统目标函数的设计[61,63,64,65]。

**本文结构：** 为了把这些变化写清楚，本文不按论文逐篇罗列，而是按照技术机制构建 taxonomy。第 2 节给出总体图谱与统一任务视角；第 3 节分析推荐与广告；第 4 节分析检索、排序与多模态访问；第 5 节分析 Search Agents、RAG 与推理；第 6-7 节讨论评测、公平与基准缺口；第 8-10 节总结跨主题趋势、开放问题与结论。

## 2. 研究图谱与统一任务视角

### 2.1 主题分布

从这 68 篇论文出发，可以把整体研究图景划分为五个主题块，如表 1 所示。推荐与广告仍然是最大板块，但其内部主轴已经明显转向生成式推荐和 LLM-enhanced recommendation[4,16,21,27,28]；检索、排序与多模态访问构成第二大板块，且越来越明显地承担“证据层”的职责[31,34,39,44]；Search Agents、RAG 与推理构成第三条主线，集中讨论多步搜索、query refinement、reasoning routing 和 uncertainty-aware retrieval[46,54,56,60]；评测、公平与对齐虽然篇数不多，却最集中地体现出 SIGIR 对“什么才算好系统”的重新定义[61,62,63,65]。

**表 1 主题分布**

| 主题 | 数量 | 核心问题 | 代表工作 |
|---|---:|---|---|
| 推荐与广告 | 28 | 在 LLM、生成式建模和工业部署约束下，推荐应如何重构？ | [4], [7], [23], [27], [28] |
| 检索、排序与多模态访问 | 17 | 如何提升跨文本、视觉、代码和混合模态场景中的检索质量、效率与 grounding？ | [29], [31], [34], [39], [44] |
| Search Agents、RAG 与推理 | 15 | 搜索系统应如何在多步轨迹中规划、检索、修正并验证证据？ | [46], [54], [56], [59], [60] |
| 评测、公平与对齐 | 5 | 在 LLM 介入信息访问后，如何重新定义评测、对齐与公平？ | [61], [62], [63], [65] |
| 图结构、表示与一般性 IR | 3 | 除四大前沿之外，哪些工作仍具有一般性方法意义？ | [66], [67], [68] |

**从图谱上可以先读出三点：** （1）**推荐是最大板块**，但其主轴已从传统协同过滤转向生成、语义接口与控制；（2）**检索仍是中轴层**，只是它服务的对象从排序结果页扩展到多模态证据消费与生成系统；（3）**Search Agent 与评测治理正在上移**，它们不再只是附属话题，而是在重新规定系统应当如何被训练和判断。

### 2.2 一个统一的信息访问公式

为了把推荐、检索与 agentic search 放到同一视角下，本文把信息访问系统抽象为一个带约束的序列决策过程。给定用户状态
$$
s_t = (u, c_t, h_t),
$$
其中 $u$ 表示用户偏好或信息需求，$c_t$ 表示当前上下文，$h_t$ 表示到第 $t$ 步为止的交互历史，系统需要从检索、排序、生成、澄清、停止等动作集合中选择动作 $a_t$，同时选取证据集合 $E_t$，以最大化整体效用：
$$
\max_{\pi} \ \mathbb{E}\left[\sum_{t=1}^{T} U(y_t, E_t, a_t \mid u) - \lambda C(E_t, a_t) - \mu R_{\text{align}}(y_t, E_t, a_t)\right].
$$
这里，$U$ 代表与用户满意度、相关性或任务成功相关的效用，$C$ 代表检索次数、向量预算、延迟和上下文长度等代价，$R_{\text{align}}$ 代表公平性、可信性、可验证性和安全性等风险项。这个公式有一个直接的解释：推荐论文主要在重写 $y_t$ 的表示方式和生成方式[4,6,27,28]；检索论文主要在改善 $E_t$ 的质量、压缩方式与可定位性[29,31,39,41]；Agent/RAG 论文主要在学习策略 $\pi$，决定何时检索、如何改写查询以及何时继续推理[46,54,56,57,60]；评测与公平工作则在重新定义 $U$ 和 $R_{\text{align}}$ 的估计协议[61,62,63,65]。从这个角度看，这些论文并不是若干平行赛道，而是在同一问题分解上的不同切面。

## 3. 推荐与广告：从 ID 匹配到可控生成

### 3.1 方法谱系

推荐与广告方向的论文可以进一步划分为四个子族：生成式推荐基础层、LLM 增强的序列与多模态推荐、对齐与测试时控制、以及系统与工业部署，如表 2 所示。与过去几年“图建模 vs. 序列建模”“self-supervised vs. contrastive”这类分类方式相比，这批工作更关注项目表示如何进入生成模型、推理路径是否可信、测试时是否可控、以及系统是否能在工业条件下稳定运行[2,4,7,23,28]。

**表 2 推荐与广告方向的主要方法族**

| 方法族 | 代表工作 | 核心机制 | 主要优势 | 主要局限 |
|---|---|---|---|---|
| 生成式推荐基础层 | [4], [6], [17], [27], [28] | 项目 tokenization、可微语义 ID、联合优化生成器与符号空间 | 打通 item identity 与 autoregressive generation 的接口 | 训练复杂，符号空间设计对性能敏感 |
| LLM 增强的序列/多模态推荐 | [1], [8], [9], [16], [18], [21], [22], [24] | 用 LLM、adapter、多模态推理或意图抽象强化状态表示 | 能缓解交互历史语义贫乏与冷启动问题 | 计算成本高，收益依赖外部知识质量 |
| 对齐、控制与满意度建模 | [2], [7], [13], [25], [28] | beam-aware 训练、test-time scaling、reward modeling、reason-verify | 更贴近真实决策目标和推理可靠性 | 离线指标与在线价值之间仍有 gap |
| 统一建模、系统与工业部署 | [14], [15], [23], [26] | CTR 结构优化、GPU 系统化落地、搜索推荐统一参数空间 | 把“可训练”推进到“可部署”，并探索跨任务统一建模 | 往往需要大规模工业环境验证 |

**本节主线：** 推荐方向的变化可以概括为三点：（1）**项目表示被重写**，item identity 不再是固定输入，而成为可学习、可迁移、可控制的接口；（2）**状态表示被语义化**，模型开始从行为序列转向行为-语义-视觉联合状态；（3）**优化目标被决策化**，系统越来越强调测试时控制、满意度与可验证性，而不是只优化离线排序指标。

### 3.2 项目标记化与生成接口成为核心瓶颈

最值得注意的变化是，推荐的核心瓶颈正在从“如何学习用户和项目嵌入”转向“如何把项目身份接入生成模型”。双层生成优化工作 [4] 指出 item tokenization 和生成目标不能分开优化，其 bi-level 框架把项目编码空间与自回归生成器联动起来，并使用 meta-learning 和 gradient surgery 缓解二者之间的冲突。与之相比，DIGER [6] 更进一步：它不再把 semantic ID 当成固定离散标签，而是把这一层本身变成可微优化对象，从而缓解生成式推荐中“语义空间连续、项目标识离散”之间的结构断裂。UTGRec [27] 则把这一问题推广到跨域场景，说明在生成式推荐里，item ID 已经不再是“预先给定”的索引，而是需要被学习、重构和迁移的语义接口。

这一类工作之所以重要，不是因为它们简单地把 item embedding 换了个写法，而是因为它们直指生成式推荐最根本的结构矛盾：推荐系统需要输出具体项目，而大型生成模型天然擅长输出词表中的 token。如何在具体项目、语义概念和生成 token 之间建立稳定、可组合、可泛化的映射，决定了模型能否真正从“语言理解”过渡到“可执行推荐”[4,6,17,27]。其中，DIGER [6] 可以视为这一子方向中的关键代表，因为它把过去常被当成离散后处理步骤的 semantic ID 重新放回可优化链路，从而为后续的迁移、控制和统一建模提供了更强的接口基础[6,26,27]。从这个意义上说，项目 tokenization 已经从工程细节上升为推荐建模中的一级研究问题。

### 3.3 从交互序列到富状态表示

另一个强信号是，序列推荐论文几乎都在试图修复“纯 ID 序列状态过于贫瘠”的问题。FLAME [9] 把 ensemble diversity 压缩进单网络，通过模块化专家和 mutual learning 保留多样性，同时维持推理效率。FAVE [8] 用 flow-based dynamics 刻画序列演化，把下一步行为建模为动态流而非静态匹配。AsarRec [1] 关注自监督场景下的数据增强失真问题，用自适应增强缓解噪声传播。SPRINT [22] 用全局意图池和不确定性触发式 LLM 调用，把昂贵推理保留给不确定会话，并将其蒸馏给轻量级模型。SpecTran [24] 则把频谱结构引入 adapter 设计，试图更细粒度地把 LLM 知识迁移到序列推荐。

这些方法表面上不同，背后的共同假设却非常一致：只依靠用户历史中的离散交互行为，很难恢复足够丰富的意图状态，因此需要更高层的抽象变量来补足状态空间。这些变量可以是隐式意图、偏好流、多模态线索、外部知识或 LLM 导出的语义表征[8,9,18,21,22,24]。换句话说，这一方向正在把“状态表示”从 interaction log 的简单聚合，升级为由行为、语义、视觉和推理共同构成的复合状态。

### 3.4 多模态偏好推理与跨域迁移

多模态推荐与跨域推荐也呈现出明显的“从融合到推理”转向。MLLMRec [18] 把图结构 refinement 与多模态偏好推理结合起来，不再只做 modality fusion，而是显式建模项目之间的关系与偏好线索如何共同支持推荐。[21] 进一步引入 preference optimization，把多模态输入与用户序列决策直接连接起来。LLM-EDT [16] 和跨域扩散生成工作 [11] 说明跨域迁移也不再仅靠共享 latent space，而是越来越依赖语言引导、扩散生成或双阶段 LLM 训练来桥接异构域之间的语义鸿沟。关系项目建模工作 [20] 则表明，项目关系建模正在从“是否相似”扩展到“能否替代、能否互补”这类更接近商业场景的结构关系。

这类工作共同推动了推荐问题的重新定义：模型不仅要知道“用户可能喜欢什么”，还要知道“为什么喜欢”“在什么情境下替代或互补成立”“跨域证据如何被迁移”和“多模态证据如何支撑一个具体推荐”。这也是为什么越来越多论文开始谈 preference reasoning，而不仅是 preference scoring[18,20,21]。

### 3.5 对齐、测试时控制与工业部署

推荐方向最成熟的一类新工作，是把生成式模型真正放进决策闭环。VRec [28] 建立在 reason-verify-recommend 范式之上，通过多个 verifier 对中间推理进行校验，说明生成式推荐不再满足于“给出看起来合理的解释”，而是开始要求解释链条本身具备可审查性。BEAR [2] 聚焦 beam search 带来的训练-推理错位，把测试时的解码行为纳入训练目标。[7] 则直接研究测试时扩展策略，说明推荐系统已开始系统性借鉴大模型推理阶段的 scaling 思路。个性化奖励建模工作 [64] 与满意度问卷监督工作 [25] 则分别从 personalized reward modeling 和显式满意度信号出发，把“用户真正满意什么”前置为优化对象，而不再仅依赖点击或短期行为代理。

值得注意的是，工业系统工作并没有因为生成式推荐兴起而被边缘化。HyFormer [15] 重新审视 CTR 场景中序列建模与特征交互的角色分工，SilverTorch [23] 面向大规模 GPU 推荐系统提出统一建模系统，实时竞价生成式出价工作 [14] 则把生成式决策带进广告场景。这说明这一方向并没有“脱离部署”，相反，它们正在试图把生成式建模、推理校验和工业效率约束放进同一个研究框架中[14,23,28]。

另一篇需要单独强调的是 GEMS [26]。与大多数工作只在推荐器内部做增强不同，GEMS [26] 直接尝试在统一 LLM 中联通搜索与推荐，通过梯度多子空间调优把共享用户建模能力与任务特定能力拆分出来。它的启发不只是“一个模型做两件事”，而是提示我们：搜索可以被理解为偏好与意图的获取过程，推荐则可以被理解为在同一语义空间中的决策投影。这个视角使 [26] 成为这批论文中少数真正横跨 recommendation 和 information access 两个传统边界的工作，也解释了为什么它在 SIGIR 2026 的整体图景里具有比单点指标更大的结构意义。

## 4. 检索、排序与多模态访问：从相关性估计到证据基础设施

### 4.1 方法谱系与关键转向

检索与多模态访问方向的论文，已经不再把主要精力放在“换一个更强 encoder”上。相反，它们更关心三类问题：监督信号是否干净、交互开销是否可控、以及证据是否能够被局部定位和下游复用[29,30,31,39,44]。表 3 总结了这一方向的主要方法族。

**表 3 检索、排序与多模态访问方向的主要方法族**

| 方法族 | 代表工作 | 核心机制 | 主要优势 | 主要局限 |
|---|---|---|---|---|
| 稠密检索监督改进 | [30], [35], [36] | hard negative relabeling、latent reasoning、语义感知 pooling | 提升监督质量与语义判别能力 | 仍依赖高质量 teacher 或额外推理 |
| Token 级效率与索引压缩 | [29], [32], [39] | token pruning、filtered ANN、multi-vector compression | 适合 agent/RAG 的多次调用场景 | 常需在精度与预算之间显式折中 |
| 多模态对齐与局部 grounding | [31], [37], [41], [42], [43], [45] | region-aware alignment、graph-text QA、listwise reranking | 更强的局部证据建模与跨模态可解释性 | 评测标准仍不统一 |
| 面向下游生成的检索接口 | [33], [34], [38], [44] | 为 code generation、fake news、RAG 设计专用检索器 | 更贴近端到端任务价值 | 跨任务可迁移性仍待验证 |

**本节主线：** 检索方向的主问题已经不再是“编码器还能不能更强”，而更像三个彼此耦合的子问题：（1）**监督能否更干净**；（2）**索引能否更省**；（3）**证据能否被更细粒度地定位并被下游系统稳定消费**。

### 4.2 稠密检索的瓶颈从 backbone 转向监督质量

ARHN [30] 代表了一个非常明确的趋势：稠密检索的关键问题不再只是模型容量，而是训练数据中的 hard negative 是否真的“难且正确”。它使用开源 LLM 对 hard negative 进行 answer-centric relabeling，直接针对伪负例和歧义负例造成的训练噪声。LaSER [36] 进一步表明，显式推理不一定要在测试时展开成长链条，也可以被“内化”到 latent space 中，使检索器在不增加推理成本的情况下获得更强的 reasoning-aware 表征。[35] 则说明 even pooling strategy 这样的“旧问题”，也在新的语义监督条件下被重新激活。

这几篇论文共同说明，稠密检索当前的性能瓶颈已经显著后移。系统不再只缺一个更深的 encoder，而是缺少更可靠的训练监督、更合适的语义压缩方式，以及能与复杂下游推理兼容的表示层[30,35,36]。这与 RAG 和 agentic retrieval 的兴起是同步的，因为一旦检索结果要被多轮重用，训练阶段的微小监督偏差就更容易在下游推理链里被放大。

### 4.3 Token 级效率成为一等问题

如果把这一批检索论文放在一起看，token-level efficiency 已经从部署议题变成了算法议题。Voronoi 剪枝工作 [29] 用 Voronoi cell 估计为 late interaction 中的 token pruning 提供了几何学基础，意味着剪枝不再只是经验规则，而可以被形式化为嵌入空间中的影响区域估计。多向量压缩工作 [39] 则直接面向文本、视觉文档和视频等任意模态，研究如何在固定向量预算下做 attention-guided compression。Filtered ANN benchmarking 工作 [32] 则把系统性对比带回视野，提醒我们在混合过滤条件和大规模 embedding 检索中，近似最近邻的实现细节会实质性改变整个系统的可用性。

这一方向的意义在于，它把“多向量检索是否太贵”从一个事后抱怨，转变成一个可以系统优化的研究问题。因为在 Search Agent、mixed-modal RAG 和视觉文档访问里，检索器往往会被反复调用，任何 token 级压缩收益都会被轨迹长度成倍放大[29,32,39,44]。因此，效率研究并不是附属品，而是直接决定 agentic systems 是否可落地的核心条件。

### 4.4 多模态检索从“全局对齐”走向“局部证据定位”

多模态检索最明显的变化，是研究重心从 global embedding alignment 进一步推进到细粒度证据 grounding。视觉文档 grounding 工作 [31] 利用 MLLM 的 cross-modal attention 作为局部监督信号，把“文档相关”分解为“哪些区域支持相关”。ReAlign [41] 进一步把 reasoning-guided fine-grained alignment 引入视觉文档检索，强调相关性判断应建立在可追踪的局部视觉-文本对应关系上。内容重构工作 [42]、StructAlign [43] 和多模态 listwise reranking 工作 [45] 则分别从重构、持续对齐和重排序的角度说明，多模态访问已经不满足于单个全局向量空间中的粗粒度相似度。

这一转向背后的关键原因是：当检索结果要服务 RAG、文档问答或 agentic reasoning 时，仅仅返回“相关文档”已经不够，系统还必须知道相关性的承载位置在哪里。否则，下游模型无法稳定地消费这些证据，也无法向用户解释其决策依据[31,41,44]。从这个角度看，多模态检索的下一阶段竞争点不是更强的全局对齐，而是更可靠的 fragment-level grounding。

### 4.5 检索正在被为“下游生成”重新设计

[44] 很直接地表达了这一点：检索不再是一个独立 benchmark task，而是 universal RAG 的 mixed-modal evidence interface。代码生成检索融合工作 [34] 针对代码生成场景设计层次式检索融合，ExDR [33] 在多模态假新闻检测里构建 explanation-driven dynamic retrieval，[38] 则通过 mixture-of-experts 让不同检索专家支持不同推理需求。这些论文说明，未来检索器的价值不再完全由自身的 Recall/NDCG 决定，而更多由其是否能在复杂生成任务中提供“可用、可组合、可追踪”的证据来定义[33,34,38,44]。

## 5. Search Agents、RAG 与推理：从单轮检索到轨迹级策略学习

### 5.1 轨迹而非单轮，成为新的基本单位

最强的新信号之一，是 search research 开始系统性转向 trajectory-level modeling。[46] 基于 14M+ 真实请求研究 agentic session 的意图与轨迹动态，提出 context-driven term adoption rate 来分析模型如何跨步骤复用证据。[54] 则进一步主张，Agent 检索器不应再只从人类点击日志中学习，而应从 agent 自己的执行轨迹中学习什么样的文档对后续推理最有帮助。SmartSearch [56] 通过 process reward 直接优化 query refinement 过程，而不是只根据最终答案评估整个系统。这些论文共同表明，多步搜索中的中间状态已成为研究对象本身，而不是被隐藏在最终指标后的黑盒。

**本节主线：** Search Agent 与 RAG 方向可以归纳为三个判断：（1）**基本单位变了**，研究对象从单轮 query-document 对转向多步轨迹；（2）**中间状态变重要了**，query reformulation、trajectory supervision 与 process reward 不再只是训练技巧，而是系统能力的一部分；（3）**风险管理变显式了**，不确定性、证据冲突与 reasoning route 选择已经成为方法设计的中心。

从方法论角度看，这意味着搜索问题的基本单位正在从“查询-文档对”转变为“状态-动作-证据轨迹”。在这种设定下，第一步拿到的并非最终相关文档，而是一个能够支持下一步 query reformulation、subgoal decomposition 或证据验证的中间证据状态[46,54,56]。这与传统 ad hoc retrieval 的差异非常大，也解释了为什么越来越多方法开始讨论 process reward、trajectory supervision 和 meta-cognitive monitoring[48,56]。

### 5.2 Deep Search 与个性化问答把任务本身抬高了

Deep Research 基准工作 [59] 把 deep research 定义为需要整合结构化与非结构化证据、跨表格与图像推理，并产出多模态分析报告的复杂任务。这一 benchmark 的意义在于，它把“搜索”从 finding facts 提升到 producing analytical outputs。IPQA [52] 则聚焦 personalized question answering 中的核心意图识别，强调个性化信息访问不能只看问题表面形式，还要识别用户真正希望系统优化什么。自然语言反馈学习工作 [53] 则把自然语言反馈作为学习信号，引导系统理解用户偏好与不满意之处。跨会话证据利用工作 [49] 则说明跨会话证据积累在现实场景里非常重要，尤其在风险评估这类需要持续跟踪的任务中。

这些工作共同抬高了信息访问任务的抽象层级。系统不只是要回答问题或返回列表，而是要面向长期、跨会话、个性化甚至分析型目标来管理证据[49,52,53,59]。这也是为什么 search agents 研究会自然地与 recommendation、用户建模和长期记忆问题重新汇合。

### 5.3 不确定性、路由与验证成为显式模块

随着系统链条变长，“何时继续检索”“何时停止推理”“哪一步最不确定”开始成为核心问题。R2C [60] 通过扰动多步推理链同时量化 retrieval uncertainty 和 reasoning uncertainty。[57] 探讨在 LLM-based ranking 中何时触发更昂贵的 reasoning route，说明 reasoning 本身需要 cost-aware routing，而不是总是开启。知识冲突建模工作 [50] 与多源事实核查检索工作 [55] 则分别从知识冲突和多源证据聚合角度，强调证据之间并非天然一致，系统需要显式管理冲突、冗余和来源可靠性。FedMosaic [51] 进一步把 federated setting 引入 RAG，说明未来的 evidence interface 还可能受到参数隔离、隐私与适配器约束。

这一组工作具有很强的共识：Search Agent 不是“更会搜的 RAG”，而是“会在不确定条件下管理证据与推理成本的策略系统”[50,57,60]。如果没有 uncertainty estimation、reasoning routing 和 source conflict handling，多步搜索只会把误差沿着轨迹不断放大。

### 5.4 长视频与时空 grounding 说明 agentic reasoning 已跨出文本

虽然很多 agent 研究仍以文本为主，但时空 grounding 工作 [47] 和视频推理 grounding 工作 [58] 证明了 agentic reasoning 已开始跨向时空感知任务。前者通过协同推理在长时空视频中进行定位，后者用 curriculum reinforced reasoning 把 grounding 与长视频理解深度耦合[47,58]。这说明未来的“搜索”并不局限于网页和文本库，而是会延伸到视频、视觉文档、直播内容与混合模态世界。检索、推理与 grounding 的统一问题因此变得更迫切。

## 6. 评测、公平与对齐：LLM 时代的信息访问治理回摆

### 6.1 LLM-as-a-judge 并没有自动解决评测问题

评测方向的论文规模较小，但信号非常强。[63] 证明，只给 query 的 LLM judge 容易过度标注相关性，而用 synthetic descriptions 或 narratives 显式形式化信息需求，可以显著提升判断可靠性。[65] 更进一步指出，在特定主题下，轻量级 topic-specific classifier 反而可能比通用 LLM judge 更稳定、更可控。这两篇论文共同否定了“把 judge 换成 LLM 就自动获得更好评测”的乐观叙事。

它们的重要性在于重新提醒我们：评测协议本身也是模型设计的一部分。若信息需求没有被充分形式化，judge 的失真会直接污染研究结论；如果评测器对任务结构的理解浅于被评测系统本身，那么 LLM judge 甚至会变成新的误差源[63,65]。因此，这些工作把评测重新拉回到 IR 的核心方法论问题上，而不是把它外包给更大的语言模型。

### 6.2 对话评测、公平性与个性化奖励正在融合

FACE [62] 给 conversational information access 提供了细粒度、reference-free 的 evaluator，说明对话式信息访问的评测不应只依赖单一答案匹配，而应分析系统是否覆盖了用户需求的多个 aspect。[61] 则从 provider fairness 角度明确提出，公平并不等于曝光完全均分，而应在不同 provider 需求差异下追求 equity-oriented ranking。个性化元奖励建模工作 [64] 则把对齐问题直接做进个体层面的奖励学习。这三类工作合在一起，构成了一个清晰趋势：评测、公平与对齐正在从离线审计工具，转向系统训练与部署阶段就必须显式建模的对象[61,62,64]。

## 7. 可见评测版图与基准缺口

若从整体样本反向观察当前评价体系，可以看到 SIGIR 2026 的方法创新已经明显快于评测协议创新。表 4 总结了这批论文中最常见的评价维度。

**表 4 主要评价维度与缺口**

| 评价维度 | 典型指标/问题 | 代表工作 | 当前缺口 |
|---|---|---|---|
| 排序与推荐效果 | Recall、NDCG、MRR、HitRate、CTR | [1], [15], [21], [30], [35] | 仍偏重静态离线指标，难覆盖多步决策价值 |
| 证据质量与 grounding | region relevance、evidence support、reason verification | [28], [31], [41], [44], [55] | fragment-level grounding 尚缺统一 benchmark |
| 轨迹质量与推理效率 | process reward、trajectory reuse、reasoning route | [46], [54], [56], [57], [60] | 代价、失败恢复与用户打断尚未系统纳入 |
| 效率与系统成本 | index size、vector budget、latency、GPU serving | [23], [29], [32], [39] | 跨任务、跨模态的一致效率报告仍不足 |
| 对齐、公平与 judge reliability | satisfaction、equity、judge agreement、topic sensitivity | [25], [61], [62], [63], [65] | 多数任务仍缺 end-to-end 的治理型评测协议 |

**这一节最关键的判断是：** 当前方法创新已经明显快于评测协议创新。也就是说，系统已经变成多阶段、带预算、带治理约束的复杂流程，但相当多实验仍然使用单轮、静态、低交互的旧评测框架。

这个表反映出的核心问题是：新的信息访问系统已经是多阶段、带代价、可交互的复杂系统，但很多实验仍沿用单轮静态指标。比如，对 recommendation 而言，用户满意度与解释可信度的重要性正在上升，但标准实验仍多使用短期行为代理[25,28]；对 search agents 而言，trajectory-level reward、query refinement quality 和 uncertainty calibration 至关重要，但通用 benchmark 往往很少覆盖这些变量[46,56,60]；对多模态检索而言，是否找到了“对的文档”已不够，还要问是否找到了“对的局部证据”，而这一层目前尚无统一标准[31,41,44]。因此，这些论文最值得重视的一个元信号，是评价维度本身已经成为未来方法竞争的瓶颈。

## 8. 跨主题综合：SIGIR 2026 的核心信号是什么

把四条主线并列起来看，这些论文共同指向五个跨主题趋势。第一，推荐正在成为可控生成系统，而不再只是打分函数[4,7,25,28]。其中，DIGER [6] 代表的是推荐内部符号接口的重写，而不仅是又一种项目编码方式。第二，检索正在从独立任务变为下游生成和推理的证据基础设施[31,34,39,44]。第三，搜索正在从单轮相关性估计转向轨迹级策略学习[46,54,56,60]。第四，评测与公平不再是系统完成后的附加打分，而开始反向塑造模型结构与训练目标[61,63,64,65]。第五，任务边界本身正在松动，GEMS [26] 说明搜索与推荐可能共享更深层的用户语义子空间，而不必始终被当作两个分离系统。与此同时，效率、预算与系统约束也被提升到与效果几乎同等重要的位置，尤其在 multi-vector retrieval、GPU recommendation serving 和 test-time scaling 场景中尤为明显[7,23,29,39]。

**可以把这些跨主题变化再压缩成五个关键词：** （1）**生成化**，推荐正在从匹配函数变成生成式决策；（2）**基础设施化**，检索正在成为证据组织层；（3）**轨迹化**，搜索正在围绕多步策略展开；（4）**治理内生化**，评测、公平与对齐开始进入目标函数；（5）**边界松动**，搜索、推荐与 RAG 不再是泾渭分明的独立模块。

更深一层看，这些趋势并非彼此独立。生成式推荐需要可信的偏好证据，因此会自然地借鉴 RAG 中的 verification 思路[10,28]；而 DIGER [6] 所代表的可微语义接口，又为这种证据驱动推荐提供了更稳固的项目表示基础。Search Agent 之所以强调 query refinement 和 trajectory learning，是因为检索器已成为可重复调用的中间模块，其代价和错误会沿推理链传播[46,54,56]；GEMS [26] 则更进一步提醒我们，搜索与推荐并非只能串联，也可能在统一语义子空间中共同建模。多模态检索之所以转向局部 grounding，是因为下游模型和用户都需要知道证据到底来自哪里[31,41,44]；公平与 judge reliability 之所以重新受到重视，是因为一旦系统开始自主规划和生成，评测器与训练目标的偏差会被进一步放大[61,63,65]。因此，这些论文真正显示的信号，并不是“LLM 进入了 IR”，而是信息访问系统开始围绕“证据、控制、验证、预算、治理”这些更系统层的问题重组。

## 9. 开放问题与未来方向

### 9.1 轨迹级代价建模仍然不足

尽管很多 Search Agent 工作已经开始优化 query refinement 或 reasoning routing，但对真实系统代价的建模仍然偏粗糙。延迟、token 消耗、失败恢复、用户中断、跨会话记忆维护等因素还没有被统一纳入 benchmark 和目标函数[46,56,57,60]。如果未来 Deep Research 类任务成为主流，这一缺口会更明显[59]。

### 9.2 生成式推荐仍缺乏统一的 faithful reasoning 标准

VRec [28] 是关键一步，但行业仍缺少统一协议来判断推荐解释究竟是基于真实偏好证据，还是基于语言流畅性生成出来的“看似合理解释”。同时，DIGER [6] 虽然改善了 semantic ID 的可优化性，却并不能自动保证解释链条就因此变得可信。这会直接影响生成式推荐的可信部署，尤其是在高价值或高风险场景中[6,25,28,64]。

### 9.3 多模态证据定位需要统一评测协议

从视觉文档检索到 mixed-modal RAG，多篇论文都在强调 fragment-level evidence 或 region-aware alignment[31,41,44]。但目前还缺乏像文本 IR 中 qrels 那样通用而稳定的局部证据标注与评测体系，这会限制多模态 grounding 研究的可比性。

### 9.4 LLM 评测器需要更强的方法学护栏

评测方向的这些工作已经清楚表明，judge 质量高度依赖任务形式化、主题敏感性与协议设计[63,65]。未来研究不应只继续做 prompt engineering，而应建立更标准化的 judge conditioning、agreement reporting 和 failure auditing 机制[62,63,65]。

### 9.5 效率不是附属指标，而是下一轮方法创新的边界条件

无论是 recommendation 的 test-time scaling、GPU serving，还是 retrieval 的 token pruning、多向量压缩和 filtered ANN，都说明能力提升正在越来越受制于系统预算[7,23,29,32,39]。对 agentic information access 来说，这种约束只会更强，不会更弱。

## 10. 结论

**结论一：研究重心已经迁移。** SIGIR 2026 的研究前沿已经不再围绕单一的“更好排序器”组织，而是在形成一个更完整的信息访问技术栈。推荐方向正在生成化、可控化和可验证化[4,25,28]，其中 DIGER [6] 代表的是推荐内部符号接口的深层重写；检索方向正在基础设施化、证据化和多模态化[31,39,44]；搜索方向正在 agent 化、轨迹化和策略学习化[46,54,59]，而 GEMS [26] 则代表任务边界开始从“搜索后接推荐”转向“搜索与推荐联合建模”。评测与治理则正在从外围标准走向系统内生目标[61,63,65]。

**结论二：未来竞争点已经改变。** 如果用一句话概括，SIGIR 2026 最强的信号不是“LLM for IR”，而是“面向真实交互约束的、以证据为核心的、可控制且可验证的信息访问系统”正在成为新的主流问题设定。从未来价值看，最有潜力的并不是某个单独模型族，而是那些能够把检索、生成、推理、用户建模、效率控制和治理约束联结起来的方法。换言之，未来 SIGIR 更核心的问题也许不再是“如何排得更准”，而是“如何在真实成本、真实用户和真实风险约束下，构建能够检索、推理、解释、适配并保持可信的信息访问系统”[28,44,46,59,63]。

## 参考文献

[1] Kaike Zhang, Qi Cao, Fei Sun, et al.. "AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation". arXiv:2512.14047v2, 2025. [https://arxiv.org/abs/2512.14047v2](https://arxiv.org/abs/2512.14047v2)

[2] Weiqin Yang, Bohao Wang, Zhenxiang Xu, et al.. "BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models". arXiv:2601.22925v1, 2026. [https://arxiv.org/abs/2601.22925v1](https://arxiv.org/abs/2601.22925v1)

[3] Yantao Yu, Sen Qiao, Lei Shen, et al.. "Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation". arXiv:2604.08011v1, 2026. [https://arxiv.org/abs/2604.08011v1](https://arxiv.org/abs/2604.08011v1)

[4] Yimeng Bai, Chang Liu, Yang Zhang, et al.. "Bi-Level Optimization for Generative Recommendation: Bridging Tokenization and Generation". arXiv:2510.21242v1, 2025. [https://arxiv.org/abs/2510.21242v1](https://arxiv.org/abs/2510.21242v1)

[5] Yu Zhang, Yiwen Zhang, Yi Zhang, Lei Sang. "DIAURec: Dual-Intent Space Representation Optimization for Recommendation". arXiv:2604.09087v2, 2026. [https://arxiv.org/abs/2604.09087v2](https://arxiv.org/abs/2604.09087v2)

[6] Junchen Fu, Xuri Ge, Alexandros Karatzoglou, et al.. "Differentiable Semantic ID for Generative Recommendation". arXiv:2601.19711v3, 2026. [https://arxiv.org/abs/2601.19711v3](https://arxiv.org/abs/2601.19711v3)

[7] Fuyuan Lyu, Zhentai Chen, Jingyan Jiang, et al.. "Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation". arXiv:2512.07650v1, 2025. [https://arxiv.org/abs/2512.07650v1](https://arxiv.org/abs/2512.07650v1)

[8] Ke Shi, Yao Zhang, Feng Guo, et al.. "FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation". arXiv:2604.04427v1, 2026. [https://arxiv.org/abs/2604.04427v1](https://arxiv.org/abs/2604.04427v1)

[9] WooJoo Kim, JunYoung Kim, JaeHyung Lim, et al.. "FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation". arXiv:2604.04038v1, 2026. [https://arxiv.org/abs/2604.04038v1](https://arxiv.org/abs/2604.04038v1)

[10] Jaehyun Lee, Sanghwan Jang, SeongKu Kang, Hwanjo Yu. "Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders". arXiv:2604.07825v1, 2026. [https://arxiv.org/abs/2604.07825v1](https://arxiv.org/abs/2604.07825v1)

[11] Ziang Lu, Lei Sang, Lin Mu, Yiwen Zhang. "From Clues to Generation: Language-Guided Conditional Diffusion for Cross-Domain Recommendation". arXiv:2604.05365v1, 2026. [https://arxiv.org/abs/2604.05365v1](https://arxiv.org/abs/2604.05365v1)

[12] Zhifu Wei, Yizhou Dang, Guibing Guo, et al.. "Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation". arXiv:2604.03688v1, 2026. [https://arxiv.org/abs/2604.03688v1](https://arxiv.org/abs/2604.03688v1)

[13] Yejing Wang, Shengyu Zhou, Jinyu Lu, et al.. "GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks". arXiv:2506.16114v2, 2025. [https://arxiv.org/abs/2506.16114v2](https://arxiv.org/abs/2506.16114v2)

[14] Yinqiu Huang, Hao Ma, Wenshuai Chen, et al.. "Generative Bid Shading in Real-Time Bidding Advertising". arXiv:2508.06550v2, 2025. [https://arxiv.org/abs/2508.06550v2](https://arxiv.org/abs/2508.06550v2)

[15] Yunwen Huang, Shiyong Hong, Xijun Xiao, et al.. "HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction". arXiv:2601.12681v2, 2026. [https://arxiv.org/abs/2601.12681v2](https://arxiv.org/abs/2601.12681v2)

[16] Ziwei Liu, Qidong Liu, Wanyu Wang, et al.. "LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training". arXiv:2511.19931v1, 2025. [https://arxiv.org/abs/2511.19931v1](https://arxiv.org/abs/2511.19931v1)

[17] Yifan Liu, Yaokun Liu, Zelin Li, et al.. "Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation". arXiv:2509.10468v1, 2025. [https://arxiv.org/abs/2509.10468v1](https://arxiv.org/abs/2509.10468v1)

[18] Yuzhuo Dang, Xin Zhang, Zhiqiang Pan, et al.. "MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation". arXiv:2508.15304v2, 2025. [https://arxiv.org/abs/2508.15304v2](https://arxiv.org/abs/2508.15304v2)

[19] Tongyoung Kim, Soojin Yoon, Seongku Kang, et al.. "MVIGER: Multi-View Variational Integration of Complementary Knowledge for Generative Recommender". arXiv:2408.08686v3, 2024. [https://arxiv.org/abs/2408.08686v3](https://arxiv.org/abs/2408.08686v3)

[20] Junting Wang, Chenghuan Guo, Jiao Yang, et al.. "Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items". arXiv:2507.22268v2, 2025. [https://arxiv.org/abs/2507.22268v2](https://arxiv.org/abs/2507.22268v2)

[21] Yu Wang, Yonghui Yang, Le Wu, et al.. "Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation". arXiv:2511.18740v2, 2025. [https://arxiv.org/abs/2511.18740v2](https://arxiv.org/abs/2511.18740v2)

[22] Gyuseok Lee, Wonbin Kweon, Zhenrui Yue, et al.. "SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation". arXiv:2508.00570v3, 2025. [https://arxiv.org/abs/2508.00570v3](https://arxiv.org/abs/2508.00570v3)

[23] Bi Xue, Hong Wu, Lei Chen, et al.. "SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs". arXiv:2511.14881v2, 2025. [https://arxiv.org/abs/2511.14881v2](https://arxiv.org/abs/2511.14881v2)

[24] Yu Cui, Feng Liu, Zhaoxiang Wang, et al.. "SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation". arXiv:2601.21986v1, 2026. [https://arxiv.org/abs/2601.21986v1](https://arxiv.org/abs/2601.21986v1)

[25] Na Li, Jiaqi Yu, Minzhi Xie, et al.. "Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation". arXiv:2601.20215v1, 2026. [https://arxiv.org/abs/2601.20215v1](https://arxiv.org/abs/2601.20215v1)

[26] Jujia Zhao, Zihan Wang, Shuaiqun Pan, et al.. "Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning". arXiv:2601.09496v1, 2026. [https://arxiv.org/abs/2601.09496v1](https://arxiv.org/abs/2601.09496v1)

[27] Bowen Zheng, Hongyu Lu, Yu Chen, et al.. "Universal Item Tokenization for Transferable Generative Recommendation". arXiv:2504.04405v3, 2025. [https://arxiv.org/abs/2504.04405v3](https://arxiv.org/abs/2504.04405v3)

[28] Xinyu Lin, Hanqing Zeng, Hanchao Yu, et al.. "Verifiable Reasoning for LLM-based Generative Recommendation". arXiv:2603.07725v1, 2026. [https://arxiv.org/abs/2603.07725v1](https://arxiv.org/abs/2603.07725v1)

[29] Yash Kankanampati, Yuxuan Zong, Nadi Tomeh, et al.. "A Voronoi Cell Formulation for Principled Token Pruning in Late-Interaction Retrieval Models". arXiv:2603.09933v2, 2026. [https://arxiv.org/abs/2603.09933v2](https://arxiv.org/abs/2603.09933v2)

[30] Hyewon Choi, Jooyoung Choi, Hansol Jang, et al.. "ARHN: Answer-Centric Relabeling of Hard Negatives with Open-Source LLMs for Dense Retrieval". arXiv:2604.11092v1, 2026. [https://arxiv.org/abs/2604.11092v1](https://arxiv.org/abs/2604.11092v1)

[31] Wanqing Cui, Wei Huang, Yazhi Guo, et al.. "Attention Grounded Enhancement for Visual Document Retrieval". arXiv:2511.13415v1, 2025. [https://arxiv.org/abs/2511.13415v1](https://arxiv.org/abs/2511.13415v1)

[32] Patrick Iff, Paul Bruegger, Marcin Chrapek, et al.. "Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors". arXiv:2507.21989v3, 2025. [https://arxiv.org/abs/2507.21989v3](https://arxiv.org/abs/2507.21989v3)

[33] Guoxuan Ding, Yuqing Li, Ziyan Zhou, et al.. "ExDR: Explanation-driven Dynamic Retrieval Enhancement for Multimodal Fake News Detection". arXiv:2601.15820v1, 2026. [https://arxiv.org/abs/2601.15820v1](https://arxiv.org/abs/2601.15820v1)

[34] Nikita Sorokin, Ivan Sedykh, Valentin Malykh. "Hierarchical Embedding Fusion for Retrieval-Augmented Code Generation". arXiv:2603.06593v1, 2026. [https://arxiv.org/abs/2603.06593v1](https://arxiv.org/abs/2603.06593v1)

[35] David Otero, Javier Parapar. "Hybrid Pooling with LLMs via Relevance Context Learning". arXiv:2602.08457v1, 2026. [https://arxiv.org/abs/2602.08457v1](https://arxiv.org/abs/2602.08457v1)

[36] Jiajie Jin, Yanzhao Zhang, Mingxin Li, et al.. "LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval". arXiv:2603.01425v1, 2026. [https://arxiv.org/abs/2603.01425v1](https://arxiv.org/abs/2603.01425v1)

[37] Yuhang Liu, Minglai Shao, Zengyi Wo, et al.. "Learning Noise-Resilient and Transferable Graph-Text Alignment via Dynamic Quality Assessment". arXiv:2510.19384v1, 2025. [https://arxiv.org/abs/2510.19384v1](https://arxiv.org/abs/2510.19384v1)

[38] Chunyi Peng, Zhipeng Xu, Zhenghao Liu, et al.. "Mixture-of-Retrieval Experts for Reasoning-Guided Multimodal Knowledge Exploitation". arXiv:2505.22095v2, 2025. [https://arxiv.org/abs/2505.22095v2](https://arxiv.org/abs/2505.22095v2)

[39] Hanxiang Qin, Alexander Martin, Rohan Jha, et al.. "Multi-Vector Index Compression in Any Modality". arXiv:2602.21202v1, 2026. [https://arxiv.org/abs/2602.21202v1](https://arxiv.org/abs/2602.21202v1)

[40] Jiahao Zhang, Shaofei Huang, Yaxiong Wang, Zhedong Zheng. "Pretrain-then-Adapt: Uncertainty-Aware Test-Time Adaptation for Text-based Person Search". arXiv:2604.08598v1, 2026. [https://arxiv.org/abs/2604.08598v1](https://arxiv.org/abs/2604.08598v1)

[41] Hao Yang, Yifan Ji, Zhipeng Xu, et al.. "ReAlign: Optimizing the Visual Document Retriever with Reasoning-Guided Fine-Grained Alignment". arXiv:2604.07419v1, 2026. [https://arxiv.org/abs/2604.07419v1](https://arxiv.org/abs/2604.07419v1)

[42] Jiahan Chen, Da Li, Hengran Zhang, et al.. "Reconstructing Content via Collaborative Attention to Improve Multimodal Embedding Quality". arXiv:2603.01471v1, 2026. [https://arxiv.org/abs/2603.01471v1](https://arxiv.org/abs/2603.01471v1)

[43] Shaokun Wang, Weili Guan, Jizhou Han, et al.. "StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval". arXiv:2601.20597v1, 2026. [https://arxiv.org/abs/2601.20597v1](https://arxiv.org/abs/2601.20597v1)

[44] Chenghao Zhang, Guanting Dong, Xinyu Yang, Zhicheng Dou. "Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation". arXiv:2510.17354v1, 2025. [https://arxiv.org/abs/2510.17354v1](https://arxiv.org/abs/2510.17354v1)

[45] Hongyi Cai. "When Vision Meets Texts in Listwise Reranking". arXiv:2601.20623v1, 2026. [https://arxiv.org/abs/2601.20623v1](https://arxiv.org/abs/2601.20623v1)

[46] Jingjie Ning, João Coelho, Yibo Kong, et al.. "Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests". arXiv:2601.17617v2, 2026. [https://arxiv.org/abs/2601.17617v2](https://arxiv.org/abs/2601.17617v2)

[47] Heng Zhao, Yew-Soon Ong, Joey Tianyi Zhou. "Agentic Spatio-Temporal Grounding via Collaborative Reasoning". arXiv:2602.13313v1, 2026. [https://arxiv.org/abs/2602.13313v1](https://arxiv.org/abs/2602.13313v1)

[48] Zhongxiang Sun, Qipeng Wang, Weijie Yu, et al.. "Deep Search with Hierarchical Meta-Cognitive Monitoring Inspired by Cognitive Neuroscience". arXiv:2601.23188v1, 2026. [https://arxiv.org/abs/2601.23188v1](https://arxiv.org/abs/2601.23188v1)

[49] Yiran Qiao, Xiang Ao, Jing Chen, et al.. "Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment". arXiv:2601.16027v1, 2026. [https://arxiv.org/abs/2601.16027v1](https://arxiv.org/abs/2601.16027v1)

[50] Tianzhe Zhao, Jiaoyan Chen, Shuxiu Zhang, et al.. "Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method". arXiv:2604.11209v1, 2026. [https://arxiv.org/abs/2604.11209v1](https://arxiv.org/abs/2604.11209v1)

[51] Zhilin Liang, Yuxiang Wang, Zimu Zhou, et al.. "FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters". arXiv:2602.05235v1, 2026. [https://arxiv.org/abs/2602.05235v1](https://arxiv.org/abs/2602.05235v1)

[52] Jieyong Kim, Maryam Amirizaniani, Soojin Yoon, Dongha Lee. "IPQA: A Benchmark for Core Intent Identification in Personalized Question Answering". arXiv:2510.23536v1, 2025. [https://arxiv.org/abs/2510.23536v1](https://arxiv.org/abs/2510.23536v1)

[53] Alireza Salemi, Hamed Zamani. "Learning from Natural Language Feedback for Personalized Question Answering". arXiv:2508.10695v1, 2025. [https://arxiv.org/abs/2508.10695v1](https://arxiv.org/abs/2508.10695v1)

[54] Yuqi Zhou, Sunhao Dai, Changle Qu, et al.. "Learning to Retrieve from Agent Trajectories". arXiv:2604.04949v1, 2026. [https://arxiv.org/abs/2604.04949v1](https://arxiv.org/abs/2604.04949v1)

[55] Shuzhi Gong, Richard O. Sinnott, Jianzhong Qi, et al.. "Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking". arXiv:2603.00267v1, 2026. [https://arxiv.org/abs/2603.00267v1](https://arxiv.org/abs/2603.00267v1)

[56] Tongyu Wen, Guanting Dong, Zhicheng Dou. "SmartSearch: Process Reward-Guided Query Refinement for Search Agents". arXiv:2601.04888v1, 2026. [https://arxiv.org/abs/2601.04888v1](https://arxiv.org/abs/2601.04888v1)

[57] Huizhong Guo, Tianjun Wei, Dongxia Wang, et al.. "Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking". arXiv:2601.18146v1, 2026. [https://arxiv.org/abs/2601.18146v1](https://arxiv.org/abs/2601.18146v1)

[58] Houlun Chen, Xin Wang, Guangyao Li, et al.. "Think with Grounding: Curriculum Reinforced Reasoning with Video Grounding for Long Video Understanding". arXiv:2602.18702v1, 2026. [https://arxiv.org/abs/2602.18702v1](https://arxiv.org/abs/2602.18702v1)

[59] Wenxuan Liu, Zixuan Li, Long Bai, et al.. "Towards Knowledgeable Deep Research: Framework and Benchmark". arXiv:2604.07720v2, 2026. [https://arxiv.org/abs/2604.07720v2](https://arxiv.org/abs/2604.07720v2)

[60] Heydar Soudani, Hamed Zamani, Faegheh Hasibi. "Uncertainty Quantification for Retrieval-Augmented Reasoning". arXiv:2510.11483v2, 2025. [https://arxiv.org/abs/2510.11483v2](https://arxiv.org/abs/2510.11483v2)

[61] Yiteng Tu, Weihang Su, Shuguang Han, et al.. "Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs". arXiv:2602.00495v1, 2026. [https://arxiv.org/abs/2602.00495v1](https://arxiv.org/abs/2602.00495v1)

[62] Hideaki Joko, Faegheh Hasibi. "FACE: A Fine-Grained Reference-Free Evaluator for Conversational Information Access". arXiv:2506.00314v3, 2025. [https://arxiv.org/abs/2506.00314v3](https://arxiv.org/abs/2506.00314v3)

[63] Jüri Keller, Maik Fröbe, Björn Engelmann, et al.. "Formalized Information Needs Improve Large-Language-Model Relevance Judgments". arXiv:2604.04140v1, 2026. [https://arxiv.org/abs/2604.04140v1](https://arxiv.org/abs/2604.04140v1)

[64] Hongru Cai, Yongqi Li, Tiezheng Yu, et al.. "One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment". arXiv:2601.18731v1, 2026. [https://arxiv.org/abs/2601.18731v1](https://arxiv.org/abs/2601.18731v1)

[65] Lukas Gienapp, Martin Potthast, Andrew Yates, et al.. "Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs". arXiv:2510.04633v2, 2025. [https://arxiv.org/abs/2510.04633v2](https://arxiv.org/abs/2510.04633v2)

[66] Yu Zhang, Yilong Luo, Mingyuan Ma, et al.. "Cohesive Group Discovery in Interaction Graphs under Explicit Density Constraints". arXiv:2508.04174v3, 2025. [https://arxiv.org/abs/2508.04174v3](https://arxiv.org/abs/2508.04174v3)

[67] Yifan Song, Fenglin Yu, Yihong Luo, et al.. "Mitigating Structural Overfitting: A Distribution-Aware Rectification Framework for Missing Feature Imputation". arXiv:2512.06356v3, 2025. [https://arxiv.org/abs/2512.06356v3](https://arxiv.org/abs/2512.06356v3)

[68] Jing Qi, Yuxiang Wang, Zhiyuan Yu, et al.. "Multi-Faceted Continual Knowledge Graph Embedding for Semantic-Aware Link Prediction". arXiv:2604.10947v1, 2026. [https://arxiv.org/abs/2604.10947v1](https://arxiv.org/abs/2604.10947v1)
