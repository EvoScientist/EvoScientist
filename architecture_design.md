# Memory-Augmented LLM Architecture Design
## 类DeepSeek Engram存算分离架构方案

---

## 1. 核心设计理念

### 1.1 存算分离原则
```
┌─────────────────────────────────────────────────────────────┐
│                    GPU HBM (高速计算)                        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  Attention  │   │    MoE      │   │   FFN       │       │
│  │   Layers    │   │   Experts   │   │  (动态推理)  │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
│         ↑                 ↑                                 │
│         │    检索增强      │                                 │
│         └─────────────────┼─────────────────────────────────┤
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Memory Layer Interface                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↕ PCIe (异步预取)
┌─────────────────────────────────────────────────────────────┐
│                    CPU DRAM (大容量存储)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Static Knowledge Memory Bank                 │   │
│  │    (N-gram Hash → Embedding Table, 100B+ params)    │   │
│  │                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ Unigram  │  │  Bigram  │  │ Trigram  │  ...     │   │
│  │  │  Table   │  │  Table   │  │  Table   │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 设计目标
| 目标 | 指标 | 实现方式 |
|------|------|----------|
| 静态知识存储 | 100B+ params in DRAM | N-gram hash embedding |
| 推理速度 | <3% throughput penalty | 异步PCIe预取 |
| 知识召回 | +3-5 benchmark points | Multi-head hashing |
| 长上下文 | 128K+ tokens | Memory作为外部知识库 |

---

## 2. 架构详细设计

### 2.1 整体模型结构

```python
# 架构层次示意
MemoryAugmentedTransformer(
    # Stage 1: Embedding + 早期层
    embedding: Embedding(vocab_size, hidden_dim),
    early_layers: [
        TransformerBlock(with_memory=True),   # Layer 2 插入 Memory
        TransformerBlock(with_memory=False),
    ],
    
    # Stage 2: 中间层 (MoE)
    middle_layers: [
        MoEBlock(num_experts=64, top_k=4),
        MoEBlock(num_experts=64, top_k=4),
        # ...
    ],
    
    # Stage 3: 后期层 + Memory
    late_layers: [
        TransformerBlock(with_memory=False),
        TransformerBlock(with_memory=True),   # Layer 15 插入 Memory
    ],
    
    # Memory Module
    memory_module: ConditionalMemory(
        ngram_orders=[1, 2, 3],
        num_heads=8,
        head_dim=1280,
        offload_to_dram=True
    ),
    
    # Output
    lm_head: Linear(hidden_dim, vocab_size)
)
```

### 2.2 Conditional Memory Layer 设计

```python
class ConditionalMemoryLayer(nn.Module):
    """
    核心Memory Layer实现
    基于 N-gram Hashing + Multi-Head Retrieval + Gating
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        ngram_orders: List[int] = [1, 2, 3],
        num_heads: int = 8,
        head_dim: int = 1280,
        vocab_size: int = 128000,
        offload_to_dram: bool = True
    ):
        super().__init__()
        self.ngram_orders = ngram_orders
        self.num_heads = num_heads
        
        # Multi-head hash projections
        self.hash_projections = nn.ModuleList([
            nn.Linear(hidden_dim, head_dim) 
            for _ in range(num_heads)
        ])
        
        # Gating mechanism (context-aware)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, head_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(head_dim, hidden_dim)
        
        # Memory Bank (可offload到DRAM)
        self.memory_bank = NgramMemoryBank(
            vocab_size=vocab_size,
            ngram_orders=ngram_orders,
            num_heads=num_heads,
            head_dim=head_dim,
            offload_to_dram=offload_to_dram
        )
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            input_ids: [batch, seq_len]
        Returns:
            memory_output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Compute N-gram hashes for each position
        ngram_hashes = self._compute_ngram_hashes(input_ids)  # [batch, seq_len, num_orders]
        
        # 2. Multi-head retrieval from memory bank
        retrieved = self.memory_bank.retrieve(ngram_hashes)  # [batch, seq_len, num_heads, head_dim]
        
        # 3. Aggregate across heads
        aggregated = retrieved.mean(dim=2)  # [batch, seq_len, head_dim]
        
        # 4. Context-aware gating
        gate_weights = self.gate(hidden_states)  # [batch, seq_len, head_dim]
        gated_memory = gate_weights * aggregated
        
        # 5. Project back to hidden dim
        output = self.output_proj(gated_memory)
        
        return output
    
    def _compute_ngram_hashes(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        计算多阶N-gram的hash索引
        使用 rolling hash 提高效率
        """
        # 简化实现示意
        hashes = []
        for order in self.ngram_orders:
            # 使用多个独立的hash函数
            for head in range(self.num_heads):
                hash_val = self._rolling_hash(input_ids, order, seed=head)
                hashes.append(hash_val)
        return torch.stack(hashes, dim=-1)
```

### 2.3 N-gram Memory Bank (支持DRAM Offloading)

```python
class NgramMemoryBank(nn.Module):
    """
    大规模N-gram Embedding存储
    支持DRAM offloading + 异步预取
    """
    def __init__(
        self,
        vocab_size: int,
        ngram_orders: List[int],
        num_heads: int,
        head_dim: int,
        offload_to_dram: bool = True
    ):
        super().__init__()
        self.offload_to_dram = offload_to_dram
        self.head_dim = head_dim
        
        # 计算每个n-gram order的table大小
        # 使用Product Quantization压缩
        self.pq_compressor = ProductQuantizer(
            vector_dim=head_dim,
            num_subvectors=8,  # 8个子空间
            bits_per_subvector=8  # 每个子空间256个centroid
        )
        
        # Hash tables for each n-gram order and head
        # 存储PQ codes而非完整向量
        self.hash_tables = nn.ParameterDict()
        for order in ngram_orders:
            table_size = self._estimate_table_size(vocab_size, order)
            # PQ codes: [table_size, num_subvectors]
            self.hash_tables[f'n{order}'] = nn.Parameter(
                torch.zeros(table_size, 8, dtype=torch.uint8),
                requires_grad=False
            )
        
        # Centroids for PQ (这些需要学习)
        self.centroids = nn.Parameter(
            torch.randn(256, head_dim // 8)  # 256 centroids per subspace
        )
        
        if offload_to_dram:
            self._setup_dram_offloading()
    
    def retrieve(self, ngram_hashes: torch.Tensor) -> torch.Tensor:
        """
        根据hash检索embedding
        支持batch retrieval
        """
        batch_size, seq_len, num_hashes = ngram_hashes.shape
        
        # 异步预取到GPU
        if self.offload_to_dram:
            self._async_prefetch(ngram_hashes)
        
        # 从hash table获取PQ codes
        pq_codes = self._lookup_pq_codes(ngram_hashes)  # [batch, seq_len, num_heads, num_subvectors]
        
        # 反量化重建向量
        reconstructed = self.pq_compressor.decode(pq_codes, self.centroids)
        
        return reconstructed
    
    def _async_prefetch(self, hashes: torch.Tensor):
        """
        异步PCIe预取，隐藏延迟
        """
        # 预测接下来需要的hash
        prefetch_hashes = self._predict_next_accesses(hashes)
        
        # 异步拷贝到GPU pinned memory
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(self.prefetch_stream):
            self._copy_to_gpu_async(prefetch_hashes)
```

### 2.4 Tokenizer Compression (减少词汇表冗余)

```python
class CompressedTokenizer:
    """
    NFKC规范化 + 大小写折叠
    减少23%词汇表大小
    """
    def __init__(self, base_tokenizer):
        self.base = base_tokenizer
        self.canonical_map = self._build_canonical_map()
    
    def normalize(self, text: str) -> str:
        """
        NFKC → NFD → strip accents → lowercase → whitespace collapse
        """
        import unicodedata
        
        # NFKC normalization
        text = unicodedata.normalize('NFKC', text)
        # NFD to separate accents
        text = unicodedata.normalize('NFD', text)
        # Strip combining marks (accents)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Lowercase
        text = text.lower()
        # Collapse whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[int]:
        normalized = self.normalize(text)
        return self.base.encode(normalized)
```

---

## 3. 预训练策略

### 3.1 两阶段预训练

```
Stage 1: Memory-agnostic Pretraining (0-70% training)
────────────────────────────────────────────────────
目标: 学习基础语言表示
配置:
  - Memory layer: 冻结或随机初始化
  - 主干网络: 正常训练
  - 数据: 通用文本语料

Stage 2: Memory-aware Pretraining (70-100% training)
────────────────────────────────────────────────────
目标: 学习将静态知识写入memory
配置:
  - Memory layer: 解冻，开始学习
  - Memory更新策略: 
      * 频繁出现的n-gram → 强化记忆
      * 罕见n-gram → 弱化或忽略
  - 数据: 高质量知识密集型语料 (Wikipedia, 教科书等)
```

### 3.2 Memory学习目标

```python
class MemoryLearningObjective(nn.Module):
    """
    训练Memory存储有用的静态知识
    """
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,      # 当前层表示
        memory_output: torch.Tensor,       # Memory检索结果
        target_hidden: torch.Tensor,       # 下一层目标表示
        next_token_logits: torch.Tensor    # 语言模型预测
    ):
        # Loss 1: Memory应该提供有用的信息
        # 如果memory_output能预测target，说明存储了正确知识
        memory_pred = F.linear(memory_output, self.pred_head.weight)
        knowledge_loss = F.mse_loss(memory_pred, target_hidden.detach())
        
        # Loss 2: Gating应该学会选择性使用memory
        # 动态内容 → gate close
        # 静态内容 → gate open
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(-1).mean()
        
        # Loss 3: 稀疏性 - 不要对所有内容都使用memory
        sparsity_loss = gate_weights.mean()
        
        return knowledge_loss + 0.1 * gate_entropy + 0.01 * sparsity_loss
```

### 3.3 静态 vs 动态知识分类

```python
class KnowledgeClassifier:
    """
    判断哪些知识应该存入memory
    """
    def __init__(self):
        # 静态知识特征
        self.static_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 日期格式
            r'[A-Z][a-z]+ [A-Z][a-z]+',  # 人名
            r'\b[A-Z]{2,}\b',  # 缩写
            # ... 更多模式
        ]
    
    def should_memorize(self, text: str, frequency: int) -> float:
        """
        返回应该记忆的置信度 [0, 1]
        
        高频 + 静态模式 → 高置信度
        低频 + 动态内容 → 低置信度
        """
        score = 0.0
        
        # 频率因素 (高频更重要)
        freq_score = min(frequency / 10000, 1.0)
        
        # 模式匹配
        pattern_score = 0.0
        for pattern in self.static_patterns:
            if re.search(pattern, text):
                pattern_score = 1.0
                break
        
        return 0.7 * freq_score + 0.3 * pattern_score
```

---

## 4. 推理时检索机制

### 4.1 推理流程

```
Input Text
    │
    ▼
┌─────────────────┐
│ Tokenizer       │ ← 规范化处理
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Layer 1         │     │                  │
└────────┬────────┘     │                  │
         │              │                  │
         ▼              │                  │
┌─────────────────┐     │   DRAM Memory    │
│ Layer 2 + MEM   │◄────┤   Bank (100B)    │
└────────┬────────┘     │                  │
         │              │   N-gram Hash    │
         ▼              │   Tables         │
┌─────────────────┐     │                  │
│ Layers 3-14     │     │                  │
│ (MoE)           │     └──────────────────┘
└────────┬────────┘              ▲
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Layer 15 + MEM  │──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Layers   │
└────────┬────────┘
         │
         ▼
    Predictions
```

### 4.2 推理优化策略

```python
class InferenceOptimizer:
    """
    推理时优化
    """
    def __init__(self, model):
        self.model = model
        self.kv_cache = {}
        self.memory_cache = LRUCache(size=10000)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ):
        # 1. Prefill阶段: 批量处理prompt
        past_key_values = None
        
        for layer in self.model.layers:
            # 检查memory cache
            ngram_hashes = self._compute_hashes(input_ids)
            
            # Cache hit → 直接使用
            # Cache miss → 从DRAM获取
            memory_out = self._retrieve_with_cache(
                ngram_hashes, 
                layer.memory_layer
            )
        
        # 2. Decode阶段: 增量生成
        for _ in range(max_new_tokens):
            # 只处理最后一个token
            # 利用KV cache避免重复计算
            pass
    
    def _retrieve_with_cache(self, hashes, memory_layer):
        """
        LRU缓存 + 预取
        """
        # 检查缓存
        cache_key = hashes.cpu().tolist()
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # 从DRAM获取
        result = memory_layer.retrieve(hashes)
        
        # 更新缓存
        self.memory_cache[cache_key] = result
        
        return result
```

### 4.3 动态知识处理

```python
class DynamicKnowledgeHandler:
    """
    处理动态变化的knowledge (news, 实时数据等)
    """
    def __init__(self, model):
        self.model = model
        self.external_retriever = None  # 可接RAG系统
    
    def process_dynamic_query(self, query: str, context: str):
        """
        动态知识 → 不使用memory，走attention路径
        静态知识 → 使用memory，减轻attention负担
        """
        # 1. 分类query类型
        is_dynamic = self._is_dynamic_query(query)
        
        if is_dynamic:
            # 动态查询: 关闭memory gate，依赖attention
            with torch.no_grad():
                self.model.set_memory_gate_bias(-10.0)  # 强制关闭
                output = self.model(query, context)
                self.model.reset_memory_gate_bias()
        else:
            # 静态查询: 正常使用memory
            output = self.model(query)
        
        return output
    
    def _is_dynamic_query(self, query: str) -> bool:
        """
        判断是否需要动态知识
        """
        dynamic_keywords = ['最新', '今天', '近期', 'current', 'latest', 'now']
        return any(kw in query.lower() for kw in dynamic_keywords)
```

---

## 5. 实验验证计划

### 5.1 实验阶段

| 阶段 | 目标 | 数据集 | 成功指标 |
|------|------|--------|----------|
| **Stage 1: Baseline** | 建立无memory基线 | Pile | Loss curve, PPL |
| **Stage 2: Memory Integration** | 集成memory layer | Pile + Wiki | +2-3 MMLU |
| **Stage 3: Offloading** | 测试DRAM offloading | - | <3% throughput loss |
| **Stage 4: Ablation** | 消融各组件 | MMLU, BBH | 各组件贡献 |
| **Stage 5: Scaling** | 扩展到更大模型 | 全量数据 | 对标Engram-27B |

### 5.2 消融实验设计

```python
ablation_configs = [
    # 配置1: 无memory (baseline)
    {'memory_enabled': False},
    
    # 配置2: 单层memory
    {'memory_enabled': True, 'memory_layers': [2]},
    
    # 配置3: 双层memory (完整)
    {'memory_enabled': True, 'memory_layers': [2, 15]},
    
    # 配置4: 无multi-head hashing
    {'memory_enabled': True, 'num_hash_heads': 1},
    
    # 配置5: 无gating
    {'memory_enabled': True, 'use_gating': False},
    
    # 配置6: 无tokenizer压缩
    {'memory_enabled': True, 'compress_tokenizer': False},
    
    # 配置7: PQ压缩
    {'memory_enabled': True, 'use_pq': True, 'pq_bits': 8},
]
```

### 5.3 评估基准

```python
evaluation_suite = {
    # 知识密集型任务 (memory应该帮助)
    'knowledge': [
        'MMLU',           # 多领域知识
        'TriviaQA',       # 事实问答
        'NaturalQuestions',
    ],
    
    # 推理任务 (主要依赖attention)
    'reasoning': [
        'BBH',            # 综合推理
        'GSM8K',          # 数学
        'HumanEval',      # 代码
    ],
    
    # 长上下文任务
    'long_context': [
        'NeedleInAHaystack',  # 检索能力
        'LongBench',          # 长文本理解
    ],
    
    # 效率指标
    'efficiency': [
        'throughput_tokens_per_sec',
        'memory_usage_gb',
        'latency_ms',
    ],
}
```

---

## 6. 实现路线图

### Phase 1: 核心组件 (2-3周)
- [ ] N-gram hash函数实现
- [ ] Memory bank基础结构
- [ ] Multi-head retrieval
- [ ] Gating mechanism

### Phase 2: 集成训练 (3-4周)
- [ ] 与Transformer集成
- [ ] 两阶段预训练pipeline
- [ ] Memory学习目标
- [ ] 小规模实验 (1B model)

### Phase 3: 优化部署 (2-3周)
- [ ] DRAM offloading
- [ ] 异步预取
- [ ] Product Quantization压缩
- [ ] 推理优化

### Phase 4: 扩展验证 (3-4周)
- [ ] 中等规模实验 (7B model)
- [ ] 完整评估
- [ ] 消融实验
- [ ] 论文撰写

---

## 7. 参考资源

### 核心论文
1. **DeepSeek Engram**: https://arxiv.org/abs/2601.07372
2. **RETRO**: "Improving language models by retrieving from trillions of tokens"
3. **Neural Turing Machines**: Graves et al., 2014
4. **Product Quantization**: Jégou et al., 2011

### 开源实现
- Engram GitHub: https://github.com/deepseek-ai/Engram
- RETRO: https://nn.labml.ai/transformers/retro/
- FAISS (PQ): https://github.com/facebookresearch/faiss

---

## 8. 下一步行动

1. **立即可做**: 搭建基础Transformer backbone + 简单memory layer
2. **短期目标**: 在小数据集验证memory学习有效性
3. **中期目标**: 完整两阶段预训练实验
4. **长期目标**: 扩展到10B+规模，对标Engram

需要我开始实现哪个部分？我建议从以下开始：
- A) 核心Memory Layer实现 (PyTorch)
- B) 训练pipeline搭建
- C) 小规模验证实验设计
