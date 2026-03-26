# Paper Reading Strategy

Extracted from the research-ideation methodology. Use this to guide structured paper analysis.

## 3-Level Reading Framework

### L1: Technical Reading (High effort)

**Goal:** Fully understand the method — able to reimplement it.

**Process:**
1. Read abstract + introduction for motivation and claims
2. Study methodology section in detail:
   - What is the exact formulation / algorithm?
   - What are the inputs, outputs, and intermediate representations?
   - What are the key hyperparameters and design choices?
3. Analyze experiments:
   - What baselines are compared?
   - What metrics are used and why?
   - Do ablation studies support the claimed contributions?
4. Check supplementary materials for implementation details
5. Look at the code (use `find_code.py`) to verify understanding

**When to use:** Papers you will directly build upon.

### L2: Analytical Reading (Medium effort)

**Goal:** Understand the *why* — motivation, design rationale, tradeoffs.

**Process:**
1. Read abstract + intro + conclusion first (5 min)
2. Study figures and tables — they tell the story
3. Focus on:
   - What problem does this solve, and why does it matter?
   - What is the key insight / intuition?
   - What are the design choices and why were they made?
   - How does this compare to alternative approaches?
4. Skim experiments for main results (skip ablations unless relevant)

**When to use:** Most papers in your literature survey.

### L3: Contextual Reading (Low effort)

**Goal:** Know what it is and where it fits in the landscape.

**Process:**
1. Read title + abstract + TLDR (if available from `scholar_search.py`)
2. Glance at figures
3. Read conclusion
4. Note: main contribution, year, citation count, relation to your work

**When to use:** Quick scanning, staying current with field trends.

---

## Reading Decision Tree

```
Is this paper directly related to my implementation?
├── Yes → L1 Technical Reading
└── No
    ├── Is it in my research area / related work?
    │   ├── Yes → L2 Analytical Reading
    │   └── No → L3 Contextual Reading
    └── Am I just browsing / monitoring?
        └── L3 Contextual Reading
```

---

## Key Questions to Answer for Each Paper

### Core Questions (all levels)

1. **What problem** does this paper address?
2. **What is the key contribution** (in one sentence)?
3. **How novel is this?** (Type 1-4 on the novelty scale)

### Deeper Questions (L1-L2)

4. **What is the key technical insight?**
5. **What assumptions does this approach make?**
6. **What are the limitations the authors acknowledge?**
7. **What limitations do they NOT acknowledge?**
8. **How does this relate to my research?**

### Implementation Questions (L1 only)

9. **Can I reproduce the results with the given information?**
10. **What would I need to change to adapt this to my problem?**
11. **What is the computational cost?**

---

## Novelty Classification (for Organize stage)

| Type | Description | Example |
|------|-------------|---------|
| Type 1 | Milestone — defines a new task or paradigm | Attention Is All You Need |
| Type 2 | New pipeline or data representation | BERT (pre-train + fine-tune paradigm) |
| Type 3 | New module or component | Flash Attention (efficient attention module) |
| Type 4 | Incremental improvement on existing methods | Most papers |

---

## Challenge-Insight Tree Construction

After reading multiple papers, build a mapping:

**Challenges** (technical problems):
- Extract from each paper: "What problem does it solve?"
- Group related challenges

**Insights** (solutions / techniques):
- Extract from each paper: "What technique / insight does it use?"
- Link insights to challenges they address

**Analysis:**
- Which challenges have many solutions? → Well-studied area
- Which challenges have few solutions? → Research opportunity
- Which insights are versatile (solve many challenges)? → Powerful technique
- Which insights are underexplored? → Potential for transfer

Save the tree to `/artifacts/literature-tree.md` and update as you read more papers.
