# SIGIR 2026 Survey-EvoScientist
## A Systematic Analysis Based on 68 arXiv-Public Papers

## Abstract

**Scope:** This survey provides a systematic analysis of the SIGIR 2026 full-paper landscape based on the 68 papers that were publicly available on arXiv as of 2026-04-16. **Core observation:** Rather than fitting the older view of SIGIR as a conference centered primarily on improving retrieval models, these papers reveal a more layered information access stack. Recommendation is shifting from representation learning and matching-function optimization toward generative recommendation, item tokenization, preference alignment, and verifiable reasoning [4,6,27,28]. Retrieval is expanding from standalone ranking quality toward multimodal evidence infrastructure, with a growing focus on token-level efficiency, local evidence localization, mixed-modal retrieval, and retrieval interfaces optimized for downstream generation [29,31,39,44]. Search agents and RAG are increasingly reframing search as trajectory-level policy learning, where query refinement, trajectory supervision, uncertainty estimation, and reasoning routing matter as much as first-stage retrieval [46,54,56,60]. Meanwhile, evaluation, fairness, and alignment are no longer peripheral concerns; they are becoming direct components of objective design and assessment protocols [61,63,64,65]. **Representative signals:** DIGER [6] makes semantic IDs differentiable and directly attacks the discrete-symbol bottleneck in generative recommendation, while GEMS [26] attempts to unify search and recommendation within one LLM-centered framework, suggesting that task boundaries themselves are being rewritten. **Organization:** The survey first presents a thematic map and a unified task view, then analyzes four major fronts: recommendation and advertising, retrieval and multimodal access, search agents/RAG/reasoning, and evaluation/fairness/alignment. It concludes with a discussion of evaluation gaps, cross-theme trends, and open research questions. **Overall judgment:** The strongest signal of SIGIR 2026 is not the dominance of any single model family, but the emergence of **evidence-grounded, controllable, and agentic information access** as a common paradigm [6,26,28,44,46,59].

## 1. Introduction

**Background:** These 68 papers suggest that the defining change in SIGIR 2026 is not that yet another backbone has topped a leaderboard, but that the objective functions and system boundaries of information access are being rewritten. Recommendation papers are no longer satisfied with ranking over fixed item-ID spaces; instead, they explicitly study generation, tokenization, preference reasoning, test-time control, and user-satisfaction alignment [4,7,25,27,28]. DIGER [6] is especially notable here because it turns semantic IDs from a fixed discrete mapping into a differentiable object, creating a more direct end-to-end interface between item identity and generation objectives. Retrieval papers are also moving beyond the ranker itself, placing more emphasis on how evidence is selected, compressed, localized, reorganized, and passed to downstream models [29,31,34,39,44]. Search and RAG, in turn, are moving from single-turn query handling toward multi-step trajectory modeling, where retrieval is one step in a decision chain rather than the endpoint of the system [46,48,54,56]. GEMS [26] pushes this boundary even further by suggesting that search and recommendation may simply be two projections of the same user-modeling problem in different semantic subspaces.

**Core claim:** Taken together, these papers reveal three structural shifts: (1) **recommendation is becoming generative and interface-driven**, with the center of gravity moving from matching functions toward tokenization, reasoning, and alignment [4,6,27,28]; (2) **retrieval is becoming infrastructure**, with its value increasingly defined by evidence organization, local grounding, and downstream generative support [29,31,39,44]; and (3) **search is becoming trajectory-based and agentic**, requiring systems to learn when to retrieve, how to retrieve, and when to continue reasoning [46,54,56,60]. At the same time, evaluation, fairness, and governance are no longer afterthoughts, but increasingly shape the design of objective functions themselves [61,63,64,65].

**Organization of the survey:** To make these changes clear, this survey does not enumerate papers one by one. Instead, it organizes them through a technical taxonomy. Section 2 presents the overall research map and a unified task view. Section 3 analyzes recommendation and advertising. Section 4 covers retrieval, ranking, and multimodal access. Section 5 focuses on search agents, RAG, and reasoning. Sections 6 and 7 examine evaluation, fairness, and benchmark gaps. Sections 8 to 10 synthesize cross-theme trends, open problems, and conclusions.

## 2. Research Map and a Unified Task View

### 2.1 Theme Distribution

Starting from these 68 papers, the visible landscape can be organized into five thematic blocks, as shown in Table 1. Recommendation and advertising remain the largest block, but their internal center of gravity has clearly shifted toward generative recommendation and LLM-enhanced recommendation [4,16,21,27,28]. Retrieval, ranking, and multimodal access form the second largest block, and increasingly function as the **evidence layer** of the stack [31,34,39,44]. Search agents, RAG, and reasoning form the third major line, centering on multi-step search, query refinement, reasoning routing, and uncertainty-aware retrieval [46,54,56,60]. Evaluation, fairness, and alignment involve fewer papers, but most clearly capture SIGIR’s renewed concern with what should count as a good system in the first place [61,62,63,65].

**Table 1. Theme Distribution**

| Theme | Count | Central Question | Representative Work |
|---|---:|---|---|
| Recommendation and Advertising | 28 | How should recommendation be reformulated under LLMs, generative modeling, and deployment constraints? | [4], [7], [23], [27], [28] |
| Retrieval, Ranking, and Multimodal Access | 17 | How can we improve retrieval quality, efficiency, and grounding across text, vision, code, and mixed-modal settings? | [29], [31], [34], [39], [44] |
| Search Agents, RAG, and Reasoning | 15 | How should search systems plan, retrieve, revise, and validate evidence over multi-step trajectories? | [46], [54], [56], [59], [60] |
| Evaluation, Fairness, and Alignment | 5 | How should evaluation, alignment, and fairness be redefined once LLMs mediate information access? | [61], [62], [63], [65] |
| Graph, Representation, and General IR | 3 | Which works remain methodologically general outside the four dominant fronts? | [66], [67], [68] |

**Three immediate takeaways emerge from this map:** (1) **recommendation is the largest visible block**, but its center has moved away from conventional collaborative filtering toward generation, semantic interfaces, and controllability; (2) **retrieval remains the mid-layer of the stack**, but the objects it serves now extend from result pages to multimodal evidence consumption and generation systems; and (3) **search agents and governance are moving upward**, no longer appearing as side topics but as forces that redefine how systems should be trained and judged.

### 2.2 A Unified Information Access Formula

To place recommendation, retrieval, and agentic search in one view, we abstract information access as a constrained sequential decision process. Let the user state be
$$
s_t = (u, c_t, h_t),
$$
where $u$ denotes user preference or information need, $c_t$ is the current context, and $h_t$ is the interaction history up to step $t$. The system must choose an action $a_t$ from a space including retrieval, ranking, generation, clarification, and stopping, while also selecting an evidence set $E_t$, so as to maximize the overall utility:
$$
\max_{\pi} \ \mathbb{E}\left[\sum_{t=1}^{T} U(y_t, E_t, a_t \mid u) - \lambda C(E_t, a_t) - \mu R_{\text{align}}(y_t, E_t, a_t)\right].
$$
Here, $U$ denotes utility associated with user satisfaction, relevance, or task success; $C$ denotes costs such as retrieval calls, vector budgets, latency, and context length; and $R_{\text{align}}$ captures risks tied to fairness, trustworthiness, verifiability, and safety. This formula offers a direct interpretation: recommendation papers mainly rewrite the representation and generation of $y_t$ [4,6,27,28]; retrieval papers mainly improve the quality, compression, and localizability of $E_t$ [29,31,39,41]; agent/RAG papers mainly learn the policy $\pi$, deciding when to retrieve, how to reformulate, and when to continue reasoning [46,54,56,57,60]; and evaluation/fairness papers redefine the estimation of $U$ and $R_{\text{align}}$ [61,62,63,65]. Under this view, these papers are not parallel silos, but different cuts through the same problem decomposition.

## 3. Recommendation and Advertising: From ID Matching to Controllable Generation

### 3.1 Method Families

The recommendation and advertising papers can be further divided into four subfamilies: generative recommendation foundations, LLM-enhanced sequential and multimodal recommendation, alignment and test-time control, and systems/industrial deployment, as shown in Table 2. Compared with older taxonomies such as “graph-based vs. sequential” or “self-supervised vs. contrastive,” these works pay much more attention to how item identities enter generative models, whether the reasoning path is trustworthy, whether inference is controllable, and whether the system can operate stably under industrial constraints [2,4,7,23,28].

**Table 2. Main Method Families in Recommendation and Advertising**

| Family | Representative Work | Core Mechanism | Main Advantage | Main Limitation |
|---|---|---|---|---|
| Generative recommendation foundations | [4], [6], [17], [27], [28] | Item tokenization, differentiable semantic IDs, joint optimization of generators and symbolic spaces | Bridges item identity and autoregressive generation | Training is complex and sensitive to symbolic-space design |
| LLM-enhanced sequential / multimodal recommendation | [1], [8], [9], [16], [18], [21], [22], [24] | LLMs, adapters, multimodal reasoning, or intent abstraction for richer state representations | Mitigates the semantic poverty of interaction histories and cold-start issues | Computationally expensive and dependent on external knowledge quality |
| Alignment, control, and satisfaction modeling | [2], [7], [13], [25], [28] | Beam-aware training, test-time scaling, reward modeling, reason-verify pipelines | Closer to real decision objectives and reasoning reliability | Offline metrics still diverge from online value |
| Unified modeling, systems, and industrial deployment | [14], [15], [23], [26] | CTR re-design, GPU systems, shared search-recommendation parameter spaces | Moves from trainable models toward deployable systems and unified task modeling | Typically requires large-scale industrial validation |

**Main line of this section:** The evolution of recommendation can be summarized in three shifts: (1) **item representation is being rewritten**—item identity is no longer fixed input, but a learnable, transferable, and controllable interface; (2) **state representation is becoming semantic**—the field is moving from behavior sequences to joint behavior-semantic-visual states; and (3) **optimization is becoming decisional**—systems increasingly optimize for test-time control, satisfaction, and verifiability rather than only offline ranking metrics.

### 3.2 Item Tokenization and the Generative Interface as the New Bottleneck

The most important shift is that the core bottleneck in recommendation is moving from “how to learn user and item embeddings” to “how to plug item identity into a generative model.” The bi-level generative optimization work [4] argues that item tokenization and generation targets should not be optimized independently; its bi-level framework jointly optimizes the item code space and the autoregressive generator, using meta-learning and gradient surgery to alleviate conflicts between the two. DIGER [6] goes further: instead of treating semantic IDs as fixed discrete labels, it makes that layer itself differentiable, thereby reducing the structural break between continuous semantic spaces and discrete item identifiers in generative recommendation. UTGRec [27] extends the issue to cross-domain settings, showing that in generative recommendation, item IDs are no longer simply predefined indices but semantic interfaces that must be learned, reconstructed, and transferred.

These works matter not because they merely rename item embeddings, but because they target the most fundamental contradiction in generative recommendation: recommendation systems must output concrete items, while large generative models are naturally designed to emit tokens from a vocabulary. The question of how to establish stable, compositional, and generalizable mappings among concrete items, semantic concepts, and generative tokens determines whether the model can truly move from “language understanding” to “actionable recommendation” [4,6,17,27]. DIGER [6] is particularly representative here because it puts semantic IDs—often treated as a discrete post-processing layer—back onto the optimization path, offering a stronger interface for later transfer, control, and unified modeling [6,26,27]. In this sense, item tokenization has become a first-order research problem rather than an implementation detail.

### 3.3 From Interaction Sequences to Rich State Representations

Another strong signal is that sequential recommendation papers are almost uniformly trying to repair the poverty of pure ID-based histories. FLAME [9] compresses ensemble diversity into a single model, preserving diversity through modular experts and mutual learning while maintaining inference efficiency. FAVE [8] models sequence evolution through flow-based dynamics rather than static matching. AsarRec [1] targets augmentation distortion in self-supervised recommendation, using adaptive augmentation to reduce noise propagation. SPRINT [22] introduces a global intent pool and uncertainty-triggered LLM invocation, reserving expensive reasoning for uncertain sessions and distilling it into a lightweight predictor. SpecTran [24] brings spectral structure into adapter design to transfer LLM knowledge more finely into sequential recommendation.

Although these methods differ in form, their shared assumption is clear: interaction logs alone are too semantically weak to recover sufficiently rich intent states. Higher-level abstraction variables are therefore needed to enrich the state space. These variables may take the form of latent intents, preference flows, multimodal cues, external knowledge, or LLM-derived semantic representations [8,9,18,21,22,24]. In other words, this line of work is upgrading “state representation” from a simple aggregation of interaction logs into a composite state built from behavior, semantics, vision, and reasoning.

### 3.4 Multimodal Preference Reasoning and Cross-Domain Transfer

Multimodal and cross-domain recommendation are also shifting from fusion to reasoning. MLLMRec [18] combines graph refinement with multimodal preference reasoning, moving beyond simple modality fusion to explicitly model how item relations and preference clues jointly support recommendation. The adaptive preference optimization work [21] pushes this further by directly connecting multimodal inputs to user sequential decisions. LLM-EDT [16] and the cross-domain diffusion work [11] show that cross-domain transfer is no longer mainly about shared latent spaces, but increasingly depends on language guidance, diffusion generation, or dual-phase LLM training to bridge semantic gaps across heterogeneous domains. The relational item modeling work [20] further indicates that item relations are expanding beyond “similar or not” toward more commercially meaningful structures such as substitutability and complementarity.

Together, these works redefine recommendation as a task that must explain not only **what** a user may like, but also **why**, under which contexts substitution or complementarity holds, how cross-domain evidence transfers, and how multimodal evidence supports a concrete recommendation. This helps explain why so many recent papers now speak in terms of **preference reasoning** rather than merely preference scoring [18,20,21].

### 3.5 Alignment, Test-Time Control, and Industrial Deployment

The most mature new line in recommendation is the effort to place generative models inside an actual decision loop. VRec [28], built on the reason-verify-recommend paradigm, uses multiple verifiers to check intermediate reasoning and shows that generative recommendation is no longer content with producing merely plausible explanations; it increasingly demands that the reasoning chain itself be auditable. BEAR [2] targets the training-inference mismatch caused by beam search by incorporating test-time decoding behavior into training. The test-time scaling work [7] studies inference-time expansion directly, signaling that recommendation is beginning to systematically borrow scaling ideas from LLM inference. The personalized reward modeling work [64] and the questionnaire-based satisfaction alignment work [25] both push real user satisfaction closer to the optimization target, rather than relying only on click proxies or short-term behavior.

Systems work also remains central despite the rise of LLMs and generation. HyFormer [15] revisits the balance between sequence modeling and feature interaction in CTR scenarios. SilverTorch [23] proposes a unified large-scale recommendation system for GPUs. The generative bid-shading work [14] brings generative decision-making into advertising. These papers make it clear that this frontier is not “leaving deployment behind”; instead, it is trying to place generative modeling, reasoning verification, and industrial efficiency constraints into one integrated research frame [14,23,28].

One work that deserves special emphasis is GEMS [26]. Unlike most methods that only enhance the recommender internally, GEMS [26] directly attempts to unify search and recommendation within a single LLM by separating shared user-modeling capacity from task-specific capacity through gradient multi-subspace tuning. Its significance is not just that “one model does two tasks,” but that it invites a different view: search may be understood as preference and intent acquisition, while recommendation may be understood as decision projection within the same semantic space. This makes [26] one of the few papers in the sample that genuinely spans the traditional boundary between recommendation and information access, which is why its structural meaning is larger than its pointwise metric gains alone.

## 4. Retrieval, Ranking, and Multimodal Access: From Relevance Estimation to Evidence Infrastructure

### 4.1 Method Families and the Main Shift

Papers on retrieval and multimodal access are no longer devoting most of their effort to “finding a stronger encoder.” Instead, they increasingly revolve around three questions: can supervision be cleaner, can interaction be cheaper, and can evidence be localized and reused downstream [29,30,31,39,44]? Table 3 summarizes the main method families in this area.

**Table 3. Main Method Families in Retrieval, Ranking, and Multimodal Access**

| Family | Representative Work | Core Mechanism | Main Advantage | Main Limitation |
|---|---|---|---|---|
| Dense retrieval supervision refinement | [30], [35], [36] | Hard-negative relabeling, latent reasoning, semantically aware pooling | Improves supervision quality and semantic discrimination | Still depends on strong teachers or extra reasoning |
| Token-level efficiency and index compression | [29], [32], [39] | Token pruning, filtered ANN, multi-vector compression | Suitable for repeated calls in agent/RAG settings | Typically requires explicit accuracy-budget trade-offs |
| Multimodal alignment and local grounding | [31], [37], [41], [42], [43], [45] | Region-aware alignment, graph-text quality estimation, listwise reranking | Stronger local evidence modeling and multimodal interpretability | Evaluation standards remain unsettled |
| Retrieval interfaces for downstream generation | [33], [34], [38], [44] | Task-specific retrievers for code generation, fake news detection, and RAG | Better aligned with end-to-end task value | Cross-task transferability remains uncertain |

**Main line of this section:** The central questions in retrieval are no longer simply “can the encoder get stronger?” but rather three coupled problems: (1) **can supervision get cleaner**; (2) **can the index get cheaper**; and (3) **can evidence be localized more precisely and consumed more reliably by downstream systems**?

### 4.2 The Bottleneck in Dense Retrieval Has Shifted from the Backbone to Supervision

ARHN [30] captures a very clear trend: the key problem in dense retrieval is no longer just model capacity, but whether hard negatives are truly hard and truly negative. It uses open-source LLMs to relabel hard negatives from an answer-centric perspective, directly targeting the training noise caused by false negatives and ambiguous negatives. LaSER [36] further shows that explicit reasoning need not appear as a long chain at inference time; it can instead be internalized into latent space so that the retriever gains stronger reasoning-aware representations without paying the cost of explicit chain-of-thought generation. The relevance-context pooling work [35] shows that even classical design choices such as pooling strategies become newly important under more semantic supervision regimes.

Taken together, these papers suggest that the bottleneck in dense retrieval has moved downstream. The field no longer merely lacks a deeper encoder; it lacks cleaner supervision, better semantic compression, and representation layers that remain compatible with complex downstream reasoning [30,35,36]. This shift is synchronized with the rise of RAG and agentic retrieval, because once retrieval outputs are reused across multiple steps, even small supervision errors in training can become amplified through reasoning chains.

### 4.3 Token-Level Efficiency Has Become a First-Order Problem

If we read the retrieval papers as a whole, token-level efficiency has moved from a deployment issue to an algorithmic issue. The Voronoi pruning work [29] gives token pruning in late interaction a geometric foundation by casting it as Voronoi-cell estimation in embedding space, meaning that pruning is no longer just an empirical rule but a formalized influence estimation problem. The multi-vector compression work [39] goes further by handling arbitrary modalities, including text, visual documents, and video, under fixed vector budgets through attention-guided compression. The filtered ANN benchmarking work [32] brings systematic systems comparison back into focus, reminding us that in large-scale embedding retrieval with filtering constraints, approximate nearest-neighbor implementation details can materially change what the whole system can do.

The significance of this line is that it converts the complaint that “multi-vector retrieval is too expensive” into a research problem that can be systematically optimized. In search agents, mixed-modal RAG, and visual-document retrieval, retrievers are often called repeatedly, so even modest token-level savings scale with trajectory length [29,32,39,44]. Efficiency is therefore not a secondary issue here; it is a central condition for making agentic systems practical.

### 4.4 Multimodal Retrieval Is Moving from Global Alignment to Local Evidence Localization

The clearest change in multimodal retrieval is that the research center is shifting from global embedding alignment toward fine-grained evidence grounding. The visual document grounding work [31] uses cross-modal attention from multimodal LLMs as a local supervision signal, decomposing document relevance into the specific regions that support it. ReAlign [41] pushes this further by introducing reasoning-guided fine-grained alignment for visual document retrieval, emphasizing that relevance judgments should be grounded in traceable local visual-text correspondences. The content reconstruction work [42], StructAlign [43], and the multimodal listwise reranking work [45] likewise show from different angles that multimodal access is no longer satisfied with coarse similarity in one global embedding space.

The reason behind this shift is straightforward: once retrieval supports RAG, document QA, or agentic reasoning, it is no longer enough to return the right document; the system must also know where the relevance is grounded. Otherwise, downstream models cannot reliably consume the evidence, and users cannot be shown why the decision was made [31,41,44]. From this perspective, the next competitive frontier in multimodal retrieval is not stronger global alignment, but more reliable fragment-level grounding.

### 4.5 Retrieval Is Increasingly Being Designed for Downstream Generation

Nyx [44] expresses this shift especially clearly: retrieval is no longer a standalone benchmark task, but a mixed-modal evidence interface for universal RAG. The code-generation retrieval-fusion work [34] designs hierarchical fusion for code-generation pipelines. ExDR [33] builds explanation-driven dynamic retrieval for multimodal fake news detection. The retrieval-expert mixture work [38] uses a mixture-of-experts design to support different reasoning needs. Together, these papers suggest that the value of a retriever is no longer defined solely by its own Recall or NDCG, but increasingly by whether it can supply **usable, composable, and traceable evidence** to complex downstream generation tasks [33,34,38,44].

## 5. Search Agents, RAG, and Reasoning: From Single-Turn Retrieval to Trajectory-Level Policy Learning

### 5.1 Trajectories Rather than Single Turns Are Becoming the Basic Unit

One of the strongest signals in the sample is the move of search research toward trajectory-level modeling. The large-scale agentic search analysis [46] studies the intentions and trajectory dynamics of 14M+ real requests and introduces context-driven term adoption rate to examine how evidence is reused across steps. The trajectory-based retriever training work [54] argues that retrievers for agents should no longer learn only from human click logs, but from the agent’s own execution traces to understand which documents are useful for future reasoning. SmartSearch [56] directly optimizes query refinement through process rewards instead of evaluating the entire system only by final answers. Together, these papers show that intermediate states in multi-step search have become objects of study in their own right rather than hidden variables behind final metrics.

**Main line of this section:** The search-agent and RAG line can be summarized through three claims: (1) **the basic unit has changed**—the object of study is no longer the single query-document pair but the multi-step trajectory; (2) **intermediate states now matter**—query reformulation, trajectory supervision, and process reward are no longer mere training tricks but part of the system’s capability; and (3) **risk management has become explicit**—uncertainty, evidence conflict, and reasoning-route choice are now central design questions.

From a methodological perspective, this means that the basic unit of search is shifting from the query-document pair to the state-action-evidence trajectory. Under this view, the first retrieved document is not necessarily the final relevant result, but an intermediate evidence state that supports the next reformulation, subgoal decomposition, or verification step [46,54,56]. This is fundamentally different from classical ad hoc retrieval and helps explain why more methods now discuss process reward, trajectory supervision, and meta-cognitive monitoring [48,56].

### 5.2 Deep Search and Personalized QA Raise the Abstraction Level of the Task

The Deep Research benchmark [59] defines deep research as a task that requires integrating structured and unstructured evidence, reasoning across tables and images, and producing multimodal analytical reports. Its importance lies in raising “search” from finding facts to producing analytical outputs. IPQA [52] focuses on core intent identification in personalized question answering, arguing that personalized information access cannot be reduced to surface-form questions but must infer what the user actually wants the system to optimize. The natural-language-feedback work [53] uses user feedback as a learning signal so that the system can internalize preference and dissatisfaction more directly. The cross-session evidence work [49] further shows that accumulated evidence across sessions matters in realistic settings, especially for continuously monitored tasks such as risk assessment.

These papers collectively raise the abstraction level of information access. Systems are no longer only asked to answer questions or return ranked lists, but to manage evidence for long-term, cross-session, personalized, and analytical goals [49,52,53,59]. This is also why search-agent research is naturally reconnecting with recommendation, user modeling, and long-term memory.

### 5.3 Uncertainty, Routing, and Verification Have Become Explicit Components

As system chains become longer, questions such as “when should the system retrieve again,” “when should it stop reasoning,” and “which step is most uncertain” become central. R2C [60] perturbs multi-step reasoning chains to jointly quantify retrieval and reasoning uncertainty. The reasoning-routing work [57] studies when LLM-based ranking should invoke more expensive reasoning, showing that reasoning itself needs cost-aware routing rather than always-on use. The knowledge-conflict work [50] and the multi-source fact-checking retrieval work [55] emphasize from different angles that evidence sources are not naturally consistent; systems must explicitly manage conflict, redundancy, and source reliability. FedMosaic [51] further introduces federated constraints into RAG, suggesting that future evidence interfaces may also be shaped by parameter isolation, privacy, and adapter constraints.

The common message of these papers is strong: a search agent is not simply “a better RAG system,” but a **strategy system that manages evidence and reasoning cost under uncertainty** [50,57,60]. Without uncertainty estimation, reasoning routing, and conflict handling, multi-step search is likely to amplify error along the trajectory.

### 5.4 Long-Video and Spatio-Temporal Grounding Show That Agentic Reasoning Has Moved Beyond Text

Although much agent research still centers on text, the spatio-temporal grounding work [47] and the long-video reasoning work [58] show that agentic reasoning is already expanding into temporal and spatial perception tasks. The former performs localization in long spatio-temporal video through collaborative reasoning, while the latter couples grounding with long-video understanding through curriculum-reinforced reasoning [47,58]. This implies that future “search” will not remain confined to web pages and text corpora, but extend into video, visual documents, live-stream content, and mixed-modal environments. The unification of retrieval, reasoning, and grounding therefore becomes even more urgent.

## 6. Evaluation, Fairness, and Alignment: A Governance Turn in LLM-Mediated Information Access

### 6.1 LLM-as-a-Judge Does Not Automatically Solve Evaluation

The evaluation block is small in size but strong in signal. The information-need formalization work [63] shows that LLM judges given only the query tend to over-label relevance, whereas synthetic descriptions and narratives make judgments more reliable. The topic-specific classifier work [65] goes even further, arguing that lightweight topic-specialized classifiers may outperform prompted LLM judges in specific topical settings.

These papers jointly reject the simplistic story that “LLM-as-a-judge solves evaluation.” Instead, they bring formalized information needs, task-specific supervision, and evaluator design back to the center. Their main lesson is that evaluation protocols are themselves part of model design. If information needs are not sufficiently formalized, evaluator distortion can directly contaminate research conclusions; if the evaluator understands the task less well than the system under test, the LLM judge can become a new source of error [63,65].

### 6.2 Conversational Evaluation, Fairness, and Personalized Reward Are Converging

FACE [62] provides a fine-grained, reference-free evaluator for conversational information access, arguing that evaluation in conversational settings should not depend only on answer matching, but on whether multiple aspects of the user need are covered. The provider-fairness work [61] argues that fairness should not mean uniform exposure, but equity-oriented ranking under heterogeneous provider needs. The personalized meta-reward modeling work [64] turns alignment into an individual-level reward-learning problem. Together, these studies suggest that evaluation, fairness, and alignment are moving from offline auditing tools toward objects that must be modeled directly during training and deployment [61,62,64].

## 7. Evaluation Landscape and Benchmark Gaps

When we read the sample backward through the lens of evaluation, the clearest finding is that methodological innovation is moving faster than evaluation protocol design. Table 4 summarizes the most common evaluation dimensions in these papers.

**Table 4. Main Evaluation Dimensions and Current Gaps**

| Evaluation Dimension | Typical Metrics / Questions | Representative Work | Current Gap |
|---|---|---|---|
| Ranking and recommendation quality | Recall, NDCG, MRR, HitRate, CTR | [1], [15], [21], [30], [35] | Still dominated by static offline metrics, weak on multi-step decision value |
| Evidence quality and grounding | Region relevance, evidence support, reason verification | [28], [31], [41], [44], [55] | No unified benchmark yet for fragment-level grounding |
| Trajectory quality and reasoning efficiency | Process reward, trajectory reuse, reasoning route | [46], [54], [56], [57], [60] | Costs, recovery, and user interruption remain under-modeled |
| Efficiency and system cost | Index size, vector budget, latency, GPU serving | [23], [29], [32], [39] | Consistent cross-task, cross-modal reporting remains limited |
| Alignment, fairness, and judge reliability | Satisfaction, equity, judge agreement, topic sensitivity | [25], [61], [62], [63], [65] | Many tasks still lack end-to-end governance-aware protocols |

**The key judgment of this section is simple:** method innovation is already ahead of evaluation innovation. Information access systems have become multi-stage, budgeted, interactive, and governance-constrained, yet many experiments still rely on old single-turn and static evaluation schemes.

This mismatch surfaces in several ways. In recommendation, user satisfaction and explanation trustworthiness are becoming more important, but many standard experiments still depend on short-term behavioral proxies [25,28]. In search agents, trajectory-level reward, query-refinement quality, and uncertainty calibration are essential, yet common benchmarks often barely expose those variables [46,56,60]. In multimodal retrieval, finding the right document is no longer sufficient; the field also needs to know whether the right **local evidence** was found, and no common protocol has yet stabilized around that [31,41,44]. For this reason, one of the strongest meta-signals in these papers is that evaluation itself is becoming a bottleneck for future progress.

## 8. Cross-Theme Synthesis: What Is the Core Signal of SIGIR 2026?

Taken together, the four major lines support five cross-theme trends. First, recommendation is becoming a controllable generative system rather than merely a scoring function [4,7,25,28]. DIGER [6] is representative here because it rewrites the symbolic interface inside recommendation rather than merely proposing another item encoding. Second, retrieval is becoming the evidence infrastructure for downstream generation and reasoning [31,34,39,44]. Third, search is shifting from single-turn relevance estimation toward trajectory-level policy learning [46,54,56,60]. Fourth, evaluation and fairness are no longer attached to the system after the fact, but increasingly reshape objective functions and training targets [61,63,64,65]. Fifth, task boundaries themselves are loosening: GEMS [26] suggests that search and recommendation may share deeper user-semantic subspaces instead of remaining separate modules. At the same time, efficiency, budgets, and systems constraints are being elevated to a level nearly as important as effectiveness, especially in multi-vector retrieval, GPU recommendation serving, and test-time scaling [7,23,29,39].

**These cross-theme shifts can be compressed into five keywords:** (1) **generativization**—recommendation is moving from matching functions to generative decisions; (2) **infrastructuralization**—retrieval is becoming the evidence organization layer; (3) **trajectory-ization**—search is centering on multi-step strategies; (4) **governance internalization**—evaluation, fairness, and alignment are entering the objective function; and (5) **boundary loosening**—search, recommendation, and RAG are no longer cleanly separated modules.

At a deeper level, these trends are not independent. Generative recommendation needs trustworthy preference evidence, so it naturally borrows verification ideas from RAG [10,28]; DIGER [6], in turn, provides a stronger item interface for evidence-driven recommendation. Search agents emphasize query refinement and trajectory learning because retrievers have become repeatedly called intermediate modules whose costs and errors propagate through reasoning chains [46,54,56]; GEMS [26] then reminds us that search and recommendation need not be only pipelined, but can also be jointly modeled in a shared semantic space. Multimodal retrieval is moving toward local grounding because downstream models and users alike need to know where evidence comes from [31,41,44]. Fairness and judge reliability are being re-centered because once systems plan and generate autonomously, misalignment between evaluators and training goals becomes easier to amplify [61,63,65]. In this sense, the real signal shown by these papers is not merely that “LLMs entered IR,” but that information access systems are reorganizing themselves around **evidence, control, verification, budgets, and governance**.

## 9. Open Problems and Future Directions

### 9.1 Trajectory-Level Cost Modeling Is Still Incomplete

Although many search-agent papers now optimize query refinement or reasoning routing, real system cost modeling remains coarse. Latency, token consumption, failure recovery, user interruption, and cross-session memory maintenance have not yet been uniformly integrated into benchmarks and objectives [46,56,57,60]. This gap will become even more visible if Deep Research-style tasks become standard [59].

### 9.2 Generative Recommendation Still Lacks a Standard for Faithful Reasoning

VRec [28] is a major step, but the field still lacks a widely accepted protocol for deciding whether a recommendation rationale is genuinely grounded in user-preference evidence or merely generated as a fluent-looking explanation. DIGER [6], while improving the learnability of semantic IDs, does not automatically make explanation chains trustworthy. This issue directly affects the credible deployment of generative recommendation, especially in high-value or high-risk settings [6,25,28,64].

### 9.3 Multimodal Evidence Localization Needs a Unified Evaluation Protocol

From visual-document retrieval to mixed-modal RAG, many papers emphasize fragment-level evidence or region-aware alignment [31,41,44]. Yet the field still lacks a stable, qrels-like protocol for local evidence annotation and evaluation, which limits the comparability of multimodal grounding research.

### 9.4 LLM-Based Evaluation Needs Stronger Methodological Guardrails

These evaluation papers make it clear that judge quality depends heavily on task formalization, topic sensitivity, and protocol design [63,65]. The next step should not merely be better prompting, but more standardized procedures for judge conditioning, agreement reporting, and failure auditing [62,63,65].

### 9.5 Efficiency Is Not a Secondary Metric but a Boundary Condition for the Next Wave

Whether in recommendation via test-time scaling and GPU serving, or in retrieval via token pruning, multi-vector compression, and filtered ANN, capability improvements are increasingly constrained by system budgets [7,23,29,32,39]. For agentic information access, these constraints will only become stronger.

## 10. Conclusion

**Conclusion 1: The center of gravity has shifted.** The SIGIR 2026 frontier is no longer organized around a single question of how to build a better ranker, but around a fuller information access stack. Recommendation is becoming generative, controllable, and verifiable [4,25,28], with DIGER [6] representing a deep rewrite of the symbolic interface inside recommendation. Retrieval is becoming infrastructural, evidence-centric, and multimodal [31,39,44]. Search is becoming agentic, trajectory-based, and policy-driven [46,54,59], while GEMS [26] suggests a shift from “search followed by recommendation” toward joint search-recommendation modeling. Evaluation and governance, meanwhile, are moving from peripheral standards toward endogenous system objectives [61,63,65].

**Conclusion 2: The next competitive frontier has changed.** If one sentence had to summarize the strongest signal of SIGIR 2026, it would not be “LLMs for IR,” but rather the rise of **information access systems that are grounded in evidence, shaped by real interaction constraints, and designed to be controllable and verifiable**. The most promising future methods are therefore unlikely to be single model families in isolation; they will be methods that connect retrieval, generation, reasoning, user modeling, efficiency control, and governance constraints in one system. Put differently, the core SIGIR question may no longer be “how do we rank better?” but “how do we build information access systems that retrieve, reason, explain, adapt, and remain trustworthy under real costs, real users, and real risks?” [28,44,46,59,63]

## References

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
