# Emergent Vector Symbolic Operations in Frozen Text Embeddings: Systematic Probing Reveals Bundled Relational Superpositions and a Silent Tokenizer Defect

**Emma Leonhart**

## Abstract

Vector Symbolic Architecture (VSA) provides a mathematical framework for symbolic computation via high-dimensional vector operations: binding pairs items into dissimilar composites, while bundling superposes items into similar aggregates (Kanerva, 2009; Gayler, 2003). We systematically probe three frozen general-purpose text embedding models — mxbai-embed-large (1024-dim), nomic-embed-text (768-dim), and all-minilm (384-dim) — using Wikidata knowledge graph triples to discover which VSA-like operations these models spontaneously encode. The core finding is that frozen embeddings implement relational structure as **bundled superpositions**: for a predicate $p$, the displacement $\mathbf{d} = f(\text{tail}) - f(\text{head})$ extracts a consistent relational signal (unbundling), and $f(\text{head}) + \bar{\mathbf{d}}_p$ predicts the correct tail entity (rebundling). These operations compose: $f(\text{head}) + \bar{\mathbf{d}}_{p_1} + \bar{\mathbf{d}}_{p_2}$ predicts two-hop targets at 28.3% Hits@10. Of 159 predicates tested, 86 produce consistent displacement vectors, with 30 universal across all three models — confirming that the relational algebraic structure is a property of the semantic relationships, not of any single model. This structure corresponds to VSA bundling/unbundling rather than binding, explaining both its strengths (associative composition) and its limitations (failure on symmetric relations, where commutativity of addition causes cancellation).

We further establish a novel correspondence between the knowledge graph embedding (KGE) literature and VSA: TransE corresponds to bundling, RotatE to FHRR binding, HolE to HRR binding, and DistMult to MAP binding — a connection that has not been made in either literature despite a decade of parallel development.

The probing procedure, applied to a domain-specific seed (Engishiki, a Japanese historical text), also reveals a previously unreported defect in mxbai-embed-large: 147,687 cross-entity embedding pairs at cosine similarity ≥ 0.95, caused by WordPiece `[UNK]` token dominance. Characters with diacritical marks (ō, ū, ī, ï, ş, ṭ, etc.) are absent from the tokenizer vocabulary, replaced with `[UNK]` tokens, and when these dominate short input strings, the embedding converges to a single attractor region regardless of text content. In VSA terms, this defect destroys representational capacity: collided embeddings lose their identity vectors, making all algebraic operations — bundling, unbinding, composition — impossible. "Hokkaidō" has cosine similarity 1.0 with "Éire" and "Filasṭīn" but only 0.45 with its own ASCII equivalent "Hokkaido". This defect is silent, systemic, and invisible to standard benchmarks. The method, all code, and all data are publicly available.

## 1. Introduction

Vector Symbolic Architecture (VSA) — also known as Hyperdimensional Computing (HDC) — is a computational framework in which complex data structures are represented as high-dimensional vectors and manipulated through three core operations: **binding** (pairing items into dissimilar composites), **bundling** (superposing items into similar aggregates), and **permutation** (encoding order) (Kanerva, 2009; Gayler, 2003; Plate, 2003). The framework has been developed across multiple variants — MAP, BSC, HRR, FHRR, VTB, TPR — each implementing binding and bundling with different concrete operations (element-wise multiplication, XOR, circular convolution, etc.) but sharing the same algebraic structure (Schlegel et al., 2022; Kleyko et al., 2023).

Independently, the word2vec analogy `king - man + woman ≈ queen` (Mikolov et al., 2013) demonstrated that embedding spaces encode relational structure as vector arithmetic. TransE (Bordes et al., 2013) formalized this for knowledge graphs, training embeddings such that `h + r ≈ t`. Subsequent KGE methods — RotatE (Sun et al., 2019), HolE (Nickel et al., 2016), ComplEx (Trouillon et al., 2016), DistMult (Yang et al., 2015) — introduced different geometric operations. What has gone unnoticed is that **these KGE methods correspond directly to VSA variants**: RotatE implements FHRR binding (element-wise angle addition), HolE implements HRR binding (circular convolution), and DistMult implements MAP binding (element-wise multiplication). TransE's additive operation corresponds to VSA bundling — not binding — which explains its known inability to model symmetric relations (bundling is commutative: $A + B = B + A$). This correspondence has not been established in either literature despite a decade of parallel development.

**We systematically probe frozen general-purpose text embeddings for emergent VSA-like operations**, sweeping over all predicates in a Wikidata knowledge graph. The finding: frozen text embeddings encode relational structure as bundled superpositions. Displacement vectors extracted via subtraction (unbundling) predict missing entities and compose across multi-hop relation chains — without any relational training. The procedure is applied to three models (mxbai-embed-large, nomic-embed-text, all-minilm), finding 30 universal operations that manifest identically across architectures, confirming that the algebraic structure is substrate-independent.

The paper has three contributions:

1. **Emergent VSA bundling in frozen embeddings.** Applied to three models, the procedure discovers that relational displacement vectors function as unbundled relational signals. These signals predict missing entities (MRR up to 1.0 for functional predicates), compose across two-hop chains (28.3% Hits@10 on 5,000 tests), and are consistent across three independently trained models (30 universal operations). In VSA terms, these are bundling/unbundling operations: the result of adding a displacement to an entity is similar to the target entity, satisfying the defining property of bundling. Functional relations produce consistent bundles; symmetric relations do not — because bundling is commutative and the ± directions cancel.

2. **The KGE–VSA correspondence.** We establish that the major knowledge graph embedding methods correspond to specific VSA variants: TransE ↔ bundling, RotatE ↔ FHRR, HolE ↔ HRR, DistMult ↔ MAP. This bridges two literatures that have been developing the same mathematical structures in different vocabularies.

3. **Discovery of a silent tokenizer defect via VSA probing.** The same procedure, seeded from a Japanese historical text (Engishiki), reveals a large-scale defect in mxbai-embed-large: 147,687 cross-entity pairs at cosine ≥ 0.95 caused by WordPiece `[UNK]` token dominance. In VSA terms, this defect destroys representational capacity — collided embeddings lose their identity vectors, making all algebraic operations impossible. The defect is silent, systemic, and invisible to standard benchmarks.

### 1.1 Key Findings

1. **Frozen embeddings implement VSA bundling.** Of 159 predicates tested (≥10 triples each), 86 produce consistent displacement vectors in mxbai-embed-large, with 30 universal across all three models. The operations correspond to VSA bundling/unbundling: functional (many-to-one) relations encode as consistent displacements; symmetric relations do not — because bundling is commutative and ± directions cancel. This matches both the KGE literature (Wang et al., 2014) and VSA theory.

2. **Consistency measures signal coherence.** The correlation between geometric consistency and prediction accuracy (r = 0.861, 95% CI [0.773, 0.926]) functions as a signal-to-noise measure for the bundled relational component. Consistency is computed over all triples; MRR uses leave-one-out evaluation where each prediction excludes the test triple. The effect size between strong and moderate operations (Cohen's d = 3.092) confirms the threshold cleanly separates operations.

3. **A silent defect destroys algebraic capacity.** The procedure revealed 147,687 cross-entity embedding pairs at cosine ≥ 0.95, caused by `[UNK]` token dominance in the WordPiece tokenizer. The defect collapses all short diacritical strings — regardless of language, script, or meaning — into a single dense region. Controlled pairs confirm the mechanism: "Hokkaidō" ↔ "Éire" = 1.0 cosine, "Hokkaidō" ↔ "Hokkaido" = 0.45 cosine. In VSA terms, entities in the collapse zone have lost their identity vectors and cannot participate in any algebraic operation.

4. **The defect is silent and systemic.** Standard benchmarks (MTEB, etc.) do not test diacritic-rich input at scale. Any RAG system or semantic search using mxbai-embed-large will silently fail on queries containing diacritical marks — returning results from the `[UNK]` attractor region instead of semantically relevant results.

5. **Domain-specific seeds expose domain-specific failures.** The Engishiki seed naturally reaches romanized non-Latin terminology that standard benchmarks never touch. Different seeds probe different regions of the embedding space, and the algebraic framework provides a principled way to characterize what each region can and cannot support.

## 2. Related Work

### 2.1 Vector Symbolic Architecture and Hyperdimensional Computing

Vector Symbolic Architecture (VSA) represents structured data as high-dimensional vectors manipulated through binding, bundling, and permutation (Kanerva, 2009; Gayler, 2003). Multiple variants implement these operations differently: HRR uses circular convolution for binding (Plate, 1995; 2003), BSC uses XOR (Kanerva, 1996), MAP uses element-wise multiplication (Gayler, 2003), FHRR uses angle addition in complex space (Plate, 2003), VTB uses matrix-vector multiplication (Gosmann & Eliasmith, 2019), and TPR uses outer products (Smolensky, 1990). Schlegel et al. (2022) provide a systematic comparison; Kleyko et al. (2023) survey applications. Fong et al. (2025) formalize VSA using category theory, establishing that binding corresponds to multiplication and bundling to addition in a division ring — roles that are "mathematically distinct and cannot be interchanged."

The critical formal distinction (Plate, 2003): binding produces output *dissimilar* to its inputs ($\delta(A \otimes B, A) \approx 0$), while bundling produces output *similar* to its inputs ($\delta(A + B, A) > 0$). Our discovered operations are additive and produce results similar to their inputs — they are therefore bundling/unbundling, not binding/unbinding.

Recent work has found VSA-like structure in neural networks: McCoy et al. (2019) showed RNNs implicitly implement Tensor Product Representations, and the "Attention as Binding" analysis (2025) argues transformer attention implements soft VSA binding/unbinding. Our contribution differs: we show the *output* embedding space of frozen text models exhibits extractable relational structure consistent with VSA bundling — not the internal mechanism, but the resulting geometric properties.

### 2.2 Knowledge Graph Embedding — and Its VSA Correspondence

TransE (Bordes et al., 2013) models relations as translations ($h + r \approx t$). Subsequent methods introduced different geometric operations: RotatE (Sun et al., 2019) uses element-wise rotation in complex space; HolE (Nickel et al., 2016) uses circular correlation; ComplEx (Trouillon et al., 2016) uses complex-valued embeddings; DistMult (Yang et al., 2015) uses element-wise multiplication. Wang et al. (2014) and Kazemi & Poole (2018) analyzed which relation types each method can represent.

**We observe that these KGE methods correspond directly to VSA variants:**

| KGE Method | Scoring Function | VSA Equivalent | Operation |
|-----------|-----------------|----------------|-----------|
| TransE | $\|h + r - t\|$ | Bundling (no direct VSA binding equivalent) | Addition |
| RotatE | $\|h \circ r - t\|$ (complex) | FHRR binding | Angle addition |
| HolE | circular correlation | HRR binding | Circular convolution |
| DistMult | $h \circ r \circ t$ | MAP binding | Element-wise multiplication |
| ComplEx | $\text{Re}(h \circ r \circ \bar{t})$ | ≈ FHRR binding | Complex multiplication |

This correspondence has not been established in either literature. It explains known KGE limitations through VSA theory: TransE cannot model symmetric relations because its operation (addition) is bundling, which is commutative ($A + B = B + A$) — a well-known property of VSA bundling that makes it unsuitable for encoding asymmetric structure. Our work applies TransE-style displacement analysis to frozen embeddings but interprets the results through VSA, clarifying both what the operations are (bundling) and why they succeed or fail.

### 2.3 Word Embedding Analogies

Mikolov et al. (2013) showed that `king - man + woman ≈ queen` holds in word2vec. Subsequent work (Linzen, 2016; Rogers et al., 2017; Schluter, 2018) showed these analogies are less robust than initially claimed. Ethayarajh et al. (2019) formalized the conditions under which analogy recovery succeeds, requiring the relation to be approximately linear and low-rank. Our work is consistent: the relations we recover are those satisfying the linearity condition (functional, bijective), and those that fail are those the theory predicts will fail (symmetric, many-to-many). In VSA terms, word2vec analogies are bundling/unbundling operations — the earliest evidence of emergent VSA-like structure in trained embeddings.

### 2.4 Neurosymbolic Integration

Logic Tensor Networks (Serafini & Garcez, 2016), Neural Theorem Provers (Rocktäschel & Riedel, 2017), and DeepProbLog (Manhaeve et al., 2018) integrate logical reasoning into neural architectures. These *constructive* approaches build systems that reason logically. Our work is *observational*: we discover what algebraic structure existing spaces already encode.

### 2.5 Probing and Representation Analysis

Probing classifiers (Conneau et al., 2018; Hewitt & Manning, 2019) test what linguistic properties are encoded in learned representations. Our displacement consistency metric functions as a relational probe, but uses vector arithmetic rather than learned classifiers and sweeps over all available predicates in a knowledge base rather than testing specific hypotheses.

### 2.6 Embedding Defects and Failure Modes

The glitch token phenomenon (Li et al., 2024) documents poorly trained embeddings for low-frequency tokens in LLMs. Our collision finding extends this to sentence-embedding models, showing that entire *classes* of input (romanized non-Latin scripts, diacritical text) collapse into near-identical regions. In VSA terms, these collisions destroy representational capacity: entities that map to identical vectors cannot participate in any algebraic operation.

### 2.7 Tokenizer-Induced Information Loss

WordPiece (Schuster & Nakajima, 2012) and BPE (Sennrich et al., 2016) tokenizers are known to struggle with out-of-vocabulary and non-Latin text. Rust et al. (2021) showed that tokenizer quality strongly predicts downstream multilingual model performance. Systematic relational probing provides a way to detect these failures geometrically: by probing a specific domain via BFS traversal, tokenizer-induced information loss becomes visible as large-scale embedding collisions that destroy the algebraic structure the space otherwise supports.

## 3. Method

### 3.1 Problem Formulation

**Given:**
- An embedding function $f: \text{Text} \to \mathbb{R}^d$ (any text embedding model)
- A knowledge base $\mathcal{K} = \{(s, p, o)\}$ of subject-predicate-object triples

**Find:** The subset of predicates $P^* \subseteq P$ whose triples manifest as consistent displacement vectors in the embedding space — i.e., the predicates for which the embedding space supports VSA bundling/unbundling operations.

**Definition (Relational Displacement as Unbundling).** For a triple $(s, p, o) \in \mathcal{K}$, the *relational displacement* is:

$$\mathbf{g}_{s,p,o} = f(o) - f(s)$$

In VSA terms, this is an **unbundling** operation: if the embedding space encodes the entity $o$ as a superposition of entity-specific and relational components, subtraction of $f(s)$ isolates the relational residual. The result is similar to other relational residuals for the same predicate (a property of bundling), not dissimilar (which would characterize binding).

**Definition (Mean Relational Signal).** For a predicate $p$ with triples $\{(s_1, p, o_1), \ldots, (s_n, p, o_n)\}$, the *mean relational signal* is:

$$\bar{\mathbf{d}}_p = \frac{1}{n}\sum_{i=1}^{n} \mathbf{g}_{s_i, p, o_i}$$

**Definition (Signal Coherence).** The *signal coherence* of predicate $p$ measures the consistency of individual displacements with the mean relational signal — functioning as a signal-to-noise ratio for the bundled relational component:

$$\text{coherence}(p) = \frac{1}{n}\sum_{i=1}^{n} \cos(\mathbf{g}_{s_i,p,o_i}, \bar{\mathbf{d}}_p)$$

A predicate with coherence > 0.5 encodes as a **consistent bundled relational signal**: its triples are approximated by a single vector operation. This threshold corresponds to the standard criterion for meaningful directional agreement in high-dimensional spaces.

### 3.2 Data Pipeline: Knowledge Graph Traversal as Algebraic Probing

The key methodological choice is using **breadth-first search through an existing knowledge graph** to generate algebraic probes. This inverts the typical KGE pipeline: standard KGE methods train embeddings to encode relations; our method uses a knowledge graph to discover what VSA-like operations a frozen embedding space already supports.

BFS from a seed entity is a **directed probing strategy**: by choosing a seed in a specific domain (e.g., Engishiki, a Japanese historical text), the traversal naturally reaches the entities and terminology that are most relevant to that domain. This means the method systematically tests the embedding space's algebraic capacity in regions where it may be weakest — regions populated by obscure, non-Latin, or domain-specific terminology that standard benchmarks never touch. A seed in Japanese history pulls in romanized shrine names, historical figures with diacritical marks, and linked entities from Arabic, Irish, and indigenous-language Wikipedia articles. The choice of seed controls *which region* of the algebraic structure is tested.

1. **Entity Import.** Two seed strategies: (a) Breadth-first search from Engishiki (Q1342448), seeding 500 entities then importing all their triples and linked entities. The BFS expansion produces **34,335 unique entities** (not 500), of which 1,781 contain diacritical marks. With aliases, the total embedding count reaches 41,725. (b) Broad P31 (instance of) sampling across country-level entities to provide a domain-general baseline. Both seeds contribute to the relational displacement analysis (Section 4.1); the collision analysis (Section 5.4) focuses on the Engishiki seed because its 1,781 diacritic-bearing labels trigger tokenizer collisions at scale.

2. **Embedding.** Each entity's English label is embedded using mxbai-embed-large (1024-dim) via Ollama. Aliases receive separate embeddings. Total: 41,725 embeddings from the Engishiki seed. Labels are short text strings (typically 1-5 words), consistent with how these models are used in practice for entity linking and retrieval.

3. **Relational Displacement Computation.** For each entity-entity triple, compute the displacement vector between subject and object label embeddings. Total: 16,893 entity-entity triples across 1,472 unique predicates. This is the standard `h + r ≈ t` test from TransE, applied without training.

### 3.3 Discovery Procedure

For each predicate $p$ with $\geq 10$ entity-entity triples:

1. Compute all relational displacements $\{\mathbf{g}_i\}$
2. Compute mean displacement $\mathbf{d}_p$
3. Compute consistency: mean alignment of each $\mathbf{g}_i$ with $\mathbf{d}_p$
4. Compute pairwise consistency: mean cosine similarity between all pairs of displacements
5. Compute magnitude coefficient of variation: stability of displacement magnitudes

**Note on unit-norm embeddings.** mxbai-embed-large returns L2-normalized embeddings (||v|| = 1.0000). Consequently, displacement magnitudes are a deterministic function of cosine similarity: ||f(o) - f(s)|| = sqrt(2(1 - cos(f(o), f(s)))). The MagCV metric therefore carries no information independent of cosine distance for this model. We retain it for cross-model comparability, as other models (e.g., BioBERT) do not necessarily normalize.

### 3.4 Prediction Evaluation

For each discovered operation ($\text{consistency} > 0.5$), we evaluate prediction accuracy using **leave-one-out**:

For each triple $(s, p, o)$:
1. Compute $\mathbf{d}_{p}^{(-i)}$ = mean displacement excluding this triple
2. Predict: $\hat{\mathbf{o}} = f(s) + \mathbf{d}_{p}^{(-i)}$
3. Rank all entities by cosine similarity to $\hat{\mathbf{o}}$
4. Record the rank of the true object $o$

We report Mean Reciprocal Rank (MRR) and Hits@k for k ∈ {1, 5, 10, 50}.

### 3.5 Composition Test (Sequential Bundling)

To test whether operations compose, we find all two-hop paths $s \xrightarrow{p_1} m \xrightarrow{p_2} o$ where both $p_1$ and $p_2$ are discovered operations. We predict via **sequential bundling**:

$$\hat{\mathbf{o}} = f(s) + \bar{\mathbf{d}}_{p_1} + \bar{\mathbf{d}}_{p_2}$$

This works because VSA bundling is **associative**: $(A + B) + C = A + (B + C)$. The two relational signals can be applied in sequence without loss. We test 5,000 compositions.

## 4. Results

### 4.1 Operation Discovery

Of 159 predicates with ≥10 triples, 86 (54.1%) produce consistent displacement vectors:

| Category | Count | Alignment Range |
|----------|-------|-----------------|
| Strong operations | 32 | > 0.7 |
| Moderate operations | 54 | 0.5 – 0.7 |
| Weak/no operation | 73 | < 0.5 |

**Table 1.** Distribution of discovered operations by consistency.

The top 15 discovered operations:

| Predicate | Label | N | Alignment | Pairwise | MagCV | Cos Dist |
|-----------|-------|---|-----------|----------|-------|----------|
| P8324 | funder | 25 | 0.930 | 0.859 | 0.079 | 0.447 |
| P2633 | geography of topic | 18 | 0.910 | 0.819 | 0.097 | 0.200 |
| P9241 | demographics of topic | 21 | 0.899 | 0.799 | 0.080 | 0.215 |
| P2596 | culture | 16 | 0.896 | 0.790 | 0.063 | 0.202 |
| P5125 | Wikimedia outline | 20 | 0.887 | 0.777 | 0.089 | 0.196 |
| P7867 | category for maps | 29 | 0.878 | 0.763 | 0.099 | 0.205 |
| P8744 | economy of topic | 30 | 0.870 | 0.749 | 0.094 | 0.182 |
| P1740 | cat. for films shot here | 18 | 0.862 | 0.728 | 0.121 | 0.266 |
| P1791 | cat. for people buried here | 13 | 0.857 | 0.714 | 0.121 | 0.302 |
| P1465 | cat. for people who died here | 29 | 0.857 | 0.725 | 0.124 | 0.249 |
| P163 | flag | 31 | 0.855 | 0.723 | 0.123 | 0.208 |
| P2746 | production statistics | 11 | 0.850 | 0.696 | 0.048 | 0.411 |
| P1923 | participating team | 32 | 0.831 | 0.681 | 0.042 | 0.387 |
| P1464 | cat. for people born here | 32 | 0.814 | 0.653 | 0.145 | 0.265 |
| P237 | coat of arms | 21 | 0.798 | 0.620 | 0.138 | 0.268 |

**Table 2.** Top 15 relations by displacement consistency (alignment with mean displacement). N = number of triples. Pairwise = mean cosine similarity between all pairs of displacements. MagCV = coefficient of variation of displacement magnitudes. Cos Dist = mean cosine distance between subject and object.

### 4.2 Prediction Accuracy

Leave-one-out evaluation of all 86 discovered operations:

| Predicate | Label | N | Align | MRR | H@1 | H@10 | H@50 |
|-----------|-------|---|-------|-----|-----|------|------|
| P9241 | demographics of topic | 21 | 0.899 | 1.000 | 1.000 | 1.000 | 1.000 |
| P2596 | culture | 16 | 0.896 | 1.000 | 1.000 | 1.000 | 1.000 |
| P7867 | category for maps | 29 | 0.878 | 1.000 | 1.000 | 1.000 | 1.000 |
| P8744 | economy of topic | 30 | 0.870 | 1.000 | 1.000 | 1.000 | 1.000 |
| P5125 | Wikimedia outline | 20 | 0.887 | 0.975 | 0.950 | 1.000 | 1.000 |
| P2633 | geography of topic | 18 | 0.910 | 0.972 | 0.944 | 1.000 | 1.000 |
| P1465 | cat. for people who died here | 29 | 0.857 | 0.966 | 0.966 | 0.966 | 0.966 |
| P163 | flag | 31 | 0.855 | 0.937 | 0.903 | 0.968 | 1.000 |
| P8324 | funder | 25 | 0.930 | 0.929 | 0.920 | 0.960 | 0.960 |
| P1464 | cat. for people born here | 32 | 0.814 | 0.922 | 0.906 | 0.938 | 0.938 |
| P237 | coat of arms | 21 | 0.798 | 0.858 | 0.762 | 0.952 | 1.000 |
| P21 | sex or gender | 91 | 0.674 | 0.422 | 0.121 | 0.945 | 0.989 |
| P27 | country of citizenship | 37 | 0.690 | 0.401 | 0.162 | 0.892 | 0.973 |

**Table 3.** Prediction results for selected operations (full table in supplementary). MRR = Mean Reciprocal Rank. H@k = Hits at rank k. The four predicates achieving MRR = 1.000 are functional predicates with highly consistent Wikidata naming conventions (e.g., every country has exactly one "Demographics of [Country]" article). Perfect MRR is expected when: (a) the predicate is strictly functional (one object per subject), (b) the displacement is consistent (alignment > 0.87), and (c) the object label is semantically close to a predictable transformation of the subject. Crucially, the string overlap null model (Section 4.4) confirms this is not a string manipulation artifact: these same predicates achieve string MRR of only 0.008–0.046 vs. vector MRR of 1.000. The embedding captures the semantic operation; the label convention merely makes the target unambiguous among 41,725 candidates.

**Aggregate statistics across all 86 operations:**

| Metric | Value | 95% Bootstrap CI |
|--------|-------|-----------------|
| Mean MRR | 0.350 | — |
| Mean Hits@1 | 0.252 | — |
| Mean Hits@10 | 0.550 | — |
| Mean Hits@50 | 0.699 | — |
| Correlation (alignment ↔ MRR) | r = 0.861 | [0.773, 0.926] |
| Correlation (alignment ↔ H@1) | r = 0.848 | [0.721, 0.932] |
| Correlation (alignment ↔ H@10) | r = 0.625 | [0.469, 0.760] |
| Effect size: strong vs moderate MRR (Cohen's d) | 3.092 | (large) |

**Table 4.** Aggregate prediction statistics with bootstrap confidence intervals (10,000 resamples). All correlations survive Bonferroni correction across 3 tests (adjusted alpha = 0.017).

The correlation between displacement consistency and prediction accuracy (r = 0.861, 95% CI [0.773, 0.926]) is practically useful as a quality filter. We note that this correlation has a natural mathematical component: when displacement variance is low (high consistency), the mean displacement is by construction a better predictor. However, the correlation is not fully tautological: consistency is computed over all triples, while MRR uses **leave-one-out** evaluation where each prediction excludes the test triple, and a high-consistency predicate could still have poor MRR if the predicted region is crowded with non-target entities. The effect size between strong (>0.7) and moderate (0.5-0.7) operations is Cohen's d = 3.092, indicating the 0.7 threshold cleanly separates high-performing from marginal operations.

### 4.3 Two-Hop Composition

Over 5,000 tested two-hop compositions (S + d₁ + d₂):

| Metric | Value |
|--------|-------|
| Hits@1 | 0.058 (288/5000) |
| Hits@10 | 0.283 (1414/5000) |
| Hits@50 | 0.479 (2396/5000) |
| Mean Rank | 1029.8 |

**Table 5.** Two-hop composition results.

Selected successful compositions (Rank ≤ 5):

| Chain | Rank |
|-------|------|
| Tadahira →[citizenship]→ Japan →[history of topic]→ history of Japan | 1 |
| Tadahira →[citizenship]→ Japan →[flag]→ flag of Japan | 1 |
| Tadahira →[citizenship]→ Japan →[cat. people buried here]→ Category:Burials in Japan | 2 |
| Tadahira →[citizenship]→ Japan →[cat. people who died here]→ Category:Deaths in Japan | 2 |
| Tadahira →[citizenship]→ Japan →[cat. associated people]→ Category:Japanese people | 3 |
| Tadahira →[citizenship]→ Japan →[head of state]→ Emperor of Japan | 4 |
| Tadahira →[sex or gender]→ male →[main category]→ Category:Male | 5 |

**Table 6.** Successful two-hop compositions. Note: all examples involve Fujiwara no Tadahira because our dataset is seeded from Engishiki (Q1342448), a Japanese historical text. Tadahira is one of the most densely connected entities in this neighborhood, appearing in many two-hop paths. The composition mechanism itself is general — the examples reflect dataset composition, not a limitation of the method.

### 4.4 String Overlap Null Model

A potential concern is that the discovered displacements merely capture string-level patterns — e.g., the displacement for "history of topic" (P2184) might simply encode the string prefix "History of" rather than relational knowledge. We test this with a string overlap null model: for each triple $(s, p, o)$, we rank all entities by longest common substring ratio with the subject label. If string overlap achieves comparable MRR to vector arithmetic, the displacement is trivially explained by surface patterns.

**Result: Vector arithmetic outperforms string overlap in 39/39 tested predicates (100%).** No predicate is trivially string-based.

| Metric | Vector Arithmetic | String Overlap (LCS) | Token Overlap |
|--------|------------------|---------------------|---------------|
| Mean MRR | 0.633 | 0.013 | 0.056 |
| Predicates with MRR > 0.5 | 24 | 0 | 0 |

The gap is not marginal: mean vector MRR is 49× higher than string MRR. Even the strongest string overlap scores (max 0.093 for P163 "flag") are far below the corresponding vector MRR (0.937). The 24 predicates with vector MRR > 0.5 all have string MRR < 0.1, confirming that the embedding captures relational structure that cannot be recovered from label text alone.

**Limitations of this baseline.** The string overlap null model is deliberately simple — it tests whether vector arithmetic reduces to substring matching, not whether it outperforms all possible string-based methods. A more sophisticated baseline (e.g., regex pattern matching for predicates like "Demographics of [X]", or edit-distance heuristics) would likely close some of the gap for the most formulaic predicates. The 49× ratio should be interpreted as evidence that the displacement is not a trivial string artifact, not as a claim about the difficulty of the prediction task itself. For the most formulaic predicates (demographics-of, geography-of), the prediction is easy by any method — the interesting finding is that vector arithmetic also works for predicates without formulaic naming (flag, coat of arms, head of state).

### 4.5 Failure Analysis

Predicates that resist vector encoding:

| Predicate | Label | N | Alignment | Pattern |
|-----------|-------|---|-----------|---------|
| P3373 | sibling | 661 | 0.026 | Symmetric |
| P155 | follows | 89 | 0.050 | Sequence (variable direction) |
| P156 | followed by | 86 | 0.053 | Sequence (variable direction) |
| P1889 | different from | 222 | 0.109 | Symmetric/diverse |
| P279 | subclass of | 168 | 0.118 | Hierarchical (variable depth) |
| P26 | spouse | 138 | 0.135 | Symmetric |
| P40 | child | 254 | 0.142 | Variable direction |
| P47 | shares border with | 197 | 0.162 | Symmetric |
| P530 | diplomatic relation | 930 | 0.165 | Symmetric |
| P31 | instance of | 835 | 0.244 | Too semantically diverse |

**Table 7.** Predicates with lowest consistency. Pattern = our characterization of why the displacement is inconsistent.

Three failure modes emerge:

1. **Symmetric predicates** (sibling, spouse, shares-border-with, diplomatic-relation): No consistent displacement direction because `f(A) - f(B)` and `f(B) - f(A)` are equally valid. Alignment ≈ 0.

2. **Sequence predicates** (follows, followed-by): The displacement from "Monday" to "Tuesday" has nothing in common with the displacement from "Chapter 1" to "Chapter 2." The *relationship type* is consistent but the *direction in embedding space* is domain-dependent.

3. **Semantically overloaded predicates** (instance-of, subclass-of, part-of): "Tokyo is an instance of city" and "7 is an instance of prime number" produce wildly different displacement vectors because the predicate covers too many semantic domains.

**Instance-of (P31) at 0.244 is particularly notable.** It is the most important predicate in Wikidata (835 triples in our dataset) and a cornerstone of first-order logic, yet it does not function as a vector operation. This suggests that embedding spaces systematically under-represent relational structure: the space encodes *entities* well but *predicates* poorly.

### 4.6 Cross-Model Generalization

To test whether discovered operations are model-agnostic or artifacts of a single model's training, we ran the full pipeline on two additional embedding models: nomic-embed-text (768-dim) and all-minilm (384-dim). All three models were given identical input: the same Wikidata entities seeded from Engishiki (Q1342448) with --limit 500.

| Model | Dimensions | Embeddings | Discovered | Strong (>0.7) |
|-------|-----------|-----------|------------|---------------|
| mxbai-embed-large | 1024 | 41,725 | 86 | 32 |
| nomic-embed-text | 768 | 69,111 | 101 | 54 |
| all-minilm | 384 | 54,375 | 109 | 41 |

**Table 8.** Operations discovered per model. All three models discover operations despite different architectures and dimensionalities.

**30 operations are universal** — discovered by all three models. These include demographics-of-topic (avg alignment 0.925), culture (0.923), economy-of-topic (0.896), flag (0.883), coat of arms (0.777), and central bank (0.793). The universal operations are exclusively functional predicates, confirming the functional-vs-relational split across architectures.

| Overlap Category | Count |
|-----------------|-------|
| Found by all 3 models | 30 |
| Found by 2 models | 15 |
| Found by 1 model only | 30 |

**Table 9.** Cross-model operation overlap. 30 universal operations constitute the model-agnostic core.

Cross-model consistency correlations (alignment scores on shared predicates): mxbai vs all-minilm r = 0.779, mxbai vs nomic r = 0.554, nomic vs all-minilm r = 0.358. The positive correlations confirm that consistency is not random — predicates that work well in one model tend to work well in others, though the strength varies by model pair.

**The same algebraic structure emerges across three unrelated embedding models** with different architectures, different dimensionalities, and different training data. The discovered bundling/unbundling operations are properties of the semantic relationships themselves, not artifacts of any particular model — a hallmark of VSA's substrate-independence.

## 5. Discussion

### 5.1 Relation Types and the VSA Bundling Framework

The pattern across Tables 2 and 7 is predicted by VSA theory: **consistent bundled relational signals emerge for functional (many-to-one) and bijective (one-to-one) relations, and fail for symmetric, transitive, or many-to-many relations.** Each country has one flag, one coat of arms, one head of state — these produce consistent displacements. Symmetric relations (sibling, spouse, shares-border-with) produce no consistent direction because bundling is commutative: $f(A) - f(B)$ and $f(B) - f(A)$ point in opposite directions, and their average cancels toward zero.

This is a known property of additive superposition in all VSA variants. Binding operations (which produce dissimilar outputs) *can* encode symmetric relations — RotatE, for example, models symmetry via $r = r^{-1}$ (self-inverse rotation). But bundling cannot, because $A + B = B + A$ provides no directional signal. The fact that frozen text embeddings exhibit exactly this bundling-specific failure pattern — succeeding on functional relations, failing on symmetric ones — is evidence that the relational structure they encode is bundling, not binding.

That this pattern holds in general-purpose text embedding models — models with no relational training signal — confirms that the algebraic structure is a property of the semantic relationships themselves, not the embedding method.

### 5.2 Signal Coherence as a Quality Indicator

The r = 0.861 correlation between signal coherence and prediction accuracy is interpretable through VSA theory: coherence measures the signal-to-noise ratio of the bundled relational component. When all displacements for a predicate point roughly the same direction (high coherence), the relational signal is cleanly separable from entity-specific noise, and the mean displacement is a good predictor. When coherence is low, entity-specific variation overwhelms the relational signal.

There is a natural mathematical tendency for low-variance displacement vectors (high coherence) to produce better mean-based predictions — the correlation is therefore partly a geometric property of high-dimensional spaces, not purely an empirical discovery. What *is* empirically informative is the magnitude of the effect size between strong and moderate operations (Cohen's d = 3.092), which confirms the 0.7 threshold cleanly separates operations that work well from those that do not. The correlation is practically useful as a quality filter for identifying which predicates support reliable algebraic operations.

### 5.3 Collision Geography

We independently measure two properties of each embedding: (a) its local density (mean k-NN distance) and (b) whether it collides with a semantically distinct entity at cosine ≥ 0.95. Dense regions could in principle have few collisions if the model separates semantically distinct entities effectively even in crowded neighborhoods. The following results describe what we observe when diacritic-rich input is embedded.

### 5.4 The Embedding Collapse: Destruction of Algebraic Capacity in mxbai-embed-large

**A previously unreported defect that destroys VSA operations.** mxbai-embed-large is one of the most popular open-source embedding models, with widespread deployment in RAG systems, semantic search, and knowledge graph applications. Despite this adoption, the tokenizer defect we report here — affecting over 16,000 entities and producing 147,687 colliding embedding pairs — appears to have gone undetected. In VSA terms, this defect destroys representational capacity: entities that map to identical vectors have lost their identity, and no algebraic operation (bundling, unbundling, composition) can distinguish them. Standard embedding benchmarks (MTEB, etc.) do not systematically probe non-Latin or diacritic-rich inputs at scale; our algebraic probing procedure does, because the knowledge graph naturally reaches the obscure terminology that benchmarks miss.

**The Jinmyōchō collapse.** Our collision analysis finds 147,687 cross-entity embedding pairs with cosine similarity ≥ 0.95 that represent genuine semantic collisions: different text mapped to near-identical vectors. This count reflects *pairwise* collisions: if $k$ entities cluster together, they contribute $\binom{k}{2}$ pairs. The 147,687 total arises from approximately 16,067 entities (of 41,725) participating in at least one collision, organized into clusters of varying size. "Jinmyōchō" collides with 504 unique texts spanning romanized Japanese (kugyō, Shōtai), Arabic (Djazaïr, Filasṭīn), Irish (Éire), Brazilian indigenous languages (Aikanã, Amanayé), and IPA characters — words that share no orthographic or semantic relationship whatsoever.

**The mechanism is `[UNK]` token dominance, not diacritic stripping.** An earlier version of this paper attributed the collisions to "WordPiece diacritic stripping," but this explanation is incomplete and misleading. If the tokenizer simply stripped diacritics, "Hokkaidō" would become "Hokkaido" and "Djazaïr" would become "Djazair" — different strings that should produce different embeddings. The actual mechanism is more severe:

1. mxbai-embed-large uses a WordPiece tokenizer whose vocabulary does not include characters with diacritical marks (ō, ū, ī, ï, ş, ṭ, é, â, etc.).
2. When the tokenizer encounters these out-of-vocabulary characters, it replaces them with the `[UNK]` (unknown) token.
3. For short input strings where diacritical characters constitute a significant fraction of the content, the tokenized sequence becomes dominated by `[UNK]` tokens.
4. The BERT-based embedding model pools over this `[UNK]`-dominated sequence, producing an embedding that reflects the `[UNK]` token's learned representation rather than the actual text content.
5. **All short diacritical strings converge to the same `[UNK]`-dominated attractor region**, regardless of language, script, or meaning.

**Controlled evidence.** We embed test pairs to confirm the mechanism (full data in `collisions.csv`):

| Pair | Cosine Similarity | Interpretation |
|------|------------------|----------------|
| "Hokkaidō" ↔ "Éire" | 1.000 | Different languages, different meanings — identical embedding |
| "Jinmyōchō" ↔ "Filasṭīn" | 1.000 | Japanese ↔ Arabic — identical embedding |
| "Djazaïr" ↔ "România" | 1.000 | Arabic ↔ Romanian — identical embedding |
| "naïve" ↔ "Zürich" | 1.000 | French ↔ German — identical embedding |
| "Hokkaidō" ↔ "Hokkaido" | 0.450 | Same word, diacritic vs. ASCII — **dissimilar** |
| "Tōkyō" ↔ "Tokyo" | 0.500 | Same word, diacritic vs. ASCII — **dissimilar** |
| "Tokyo" ↔ "Berlin" | 0.751 | Control: two capitals — normal similarity |

**Table 10.** Controlled collision pairs. The diacritical version of a word is more similar to an unrelated diacritical word in a different language (cosine 1.0) than to its own ASCII equivalent (cosine ~0.45). This rules out diacritic stripping as the mechanism: if the model stripped diacritics and embedded the ASCII form, "Hokkaidō" would be close to "Hokkaido", not to "Éire". Instead, the `[UNK]` tokens overwhelm the embedding, and all `[UNK]`-dominated inputs converge to the same point.

**The collapse zone is dense, not sparse.** Geometric analysis of 16,067 colliding embeddings (vs. 74,760 non-colliding) reveals:

1. **Colliding embeddings are 2.4× denser than non-colliding ones.** Mean k-NN distance for colliding embeddings is 0.106, vs 0.258 for non-colliding (ratio 0.41×).

2. **71% of colliding embeddings fall in the densest quartile,** vs the expected 25% if uniformly distributed. Only 3.2% fall in the sparsest quartile.

3. **The collapse zone is not geometrically isolated.** The distance from a colliding embedding to its nearest non-colliding neighbor (mean 0.119) is nearly identical to the non-colliding-to-non-colliding distance (mean 0.121, ratio 0.98×).

This means the `[UNK]` attractor region sits *among* the well-structured embeddings, not apart from them. The colliding embeddings crowd into already-dense neighborhoods where the model cannot differentiate them from legitimate nearby entities.

**The defect is silent and likely exploitable.** The `[UNK]`-dominated embedding region has several concerning properties: (1) it is invisible to standard benchmarks, (2) the model returns a confident-looking embedding vector rather than an error, (3) any downstream system treating this vector as meaningful will silently produce wrong results. We hypothesize that this represents a class of silent failures that has been causing degraded performance in production systems for years — any RAG pipeline, semantic search engine, or knowledge graph application processing non-ASCII input through mxbai-embed-large has been mapping those inputs to a single undifferentiated region. The scale of affected systems is difficult to estimate, but given the model's popularity and the prevalence of diacritical marks in non-English text, the impact is likely substantial.

The phenomenon is reminiscent of glitch tokens (Li et al., 2024) but at a different scale: entire *classes of input* (any text containing diacritical marks) rather than individual tokens, and in sentence-embedding models rather than LLMs.

**Why the Engishiki seed matters.** Engishiki (Q1342448) is a 10th-century Japanese text whose entities include romanized shrine names (Jinmyōchō, Shikinaisha), historical Japanese personal names, and linked entities from Arabic, Irish, and indigenous-language Wikipedia articles. This floods the embedding space with exactly the inputs that trigger `[UNK]` token dominance, making the phenomenon measurable at scale. The defect exists regardless of seed choice — any diacritical input triggers it — but the Engishiki seed makes it *statistically visible* by providing thousands of affected entities in a single BFS traversal.

### 5.5 Practical Implications

The `[UNK]` token dominance defect has immediate practical consequences. Any system using mxbai-embed-large for semantic search, RAG, or knowledge graph completion over non-ASCII text has been silently affected. A user querying "Hokkaidō" will retrieve results from the `[UNK]` attractor region — potentially returning "Éire", "Djazaïr", or any other diacritical string — rather than results related to the Japanese island. The failure is silent: the model returns a valid-looking 1024-dimensional vector, and no error is raised.

We hypothesize that this class of silent failure extends beyond mxbai-embed-large. All three models tested use WordPiece or similar subword tokenizers, and the `[UNK]` token dominance mechanism applies to any tokenizer with incomplete character coverage. The practical recommendation is to test embedding models with diacritic-rich input before deployment, and to prefer models with byte-level tokenizers (e.g., CANINE, ByT5) or SentencePiece with full Unicode coverage for multilingual applications.

### 5.6 Limitations

1. **Three embedding models.** We validate across mxbai-embed-large (1024-dim), nomic-embed-text (768-dim), and all-minilm (384-dim), finding 30 universal relations. All three are English-language text embedding models trained on similar corpora. Testing on multilingual models or domain-specific models (e.g., biomedical) would further characterize the generality of the three-regime structure.

2. **Collision geometry analysis covers one seed.** The distance metrics characterizing the embedding collision zone (Section 5.4) are computed from the Engishiki-seeded dataset. Multi-seed analysis would test whether the same crowding pattern holds across domains.

3. **Label embeddings only.** We embed entity *labels* (short text strings), not descriptions or full articles. This deliberately mirrors how these models are used in practice for entity linking and knowledge graph completion (short query strings, not full documents). Richer textual representations might shift some entities out of the sparse zone, but the label-only setting represents a common real-world deployment pattern for these models.

4. **Potential training data overlap.** The embedding models tested were trained on large web crawls that likely include Wikipedia content, and Wikidata entities often have corresponding Wikipedia articles. This raises the possibility that some discovered displacements reflect memorized associations from training data rather than emergent geometric structure. The cross-model consistency (30 universal operations across three independently trained models) provides partial mitigation: memorization patterns would be model-specific, while consistent operations across architectures suggest structural encoding. However, a definitive test would require embedding models trained on corpora that exclude Wikipedia, which we leave for future work.

5. **Single tokenizer family.** All three models use WordPiece or similar subword tokenizers. The `[UNK]` token dominance mechanism is specific to tokenizers with incomplete character coverage. Models using byte-level tokenizers (e.g., CANINE, ByT5) or SentencePiece with full Unicode support would likely not exhibit this specific failure mode. Testing against such models would clarify the scope of the defect.

4. **Bundling, not full VSA.** We test which binary relations encode as consistent bundled superpositions (addition/subtraction). Full VSA includes binding (multiplicative operations producing dissimilar outputs), permutation (sequence encoding), and codebook structure, none of which we test. Whether frozen embeddings also support these richer VSA operations is future work.

## 6. Conclusion

We systematically probe three frozen general-purpose text embedding models for emergent Vector Symbolic Architecture (VSA) operations using Wikidata knowledge graph triples. The core finding is that frozen embeddings encode relational structure as **bundled superpositions**: displacement vectors extracted via subtraction (unbundling) predict missing entities, compose across multi-hop relation chains, and are consistent across three independently trained models (30 universal operations). The success/failure pattern — functional relations produce consistent bundles, symmetric relations do not — matches VSA theory: bundling is commutative and thus directionally uninformative for symmetric relations.

We establish a novel correspondence between the KGE and VSA literatures: TransE corresponds to bundling, RotatE to FHRR binding, HolE to HRR binding, and DistMult to MAP binding. This bridges two research communities that have been developing the same mathematical structures in different vocabularies for over a decade.

The algebraic probing procedure also reveals a silent defect in mxbai-embed-large: `[UNK]` token dominance in the WordPiece tokenizer causes 147,687 cross-entity embedding collisions, collapsing all short diacritical strings into a single attractor region regardless of language or meaning. In VSA terms, this defect destroys representational capacity — collided entities lose their identity vectors and cannot participate in any algebraic operation. The diacritical version of a word is more similar to an unrelated diacritical word in a different language (cosine 1.0) than to its own ASCII equivalent (cosine ~0.45). This defect has existed for years in a widely-deployed model, is invisible to standard benchmarks, and silently degrades any downstream system processing non-ASCII input.

The defect was discovered because the probing procedure, seeded from a Japanese historical text (Engishiki), naturally reached the diacritic-rich terminology that standard benchmarks never test. The broader lesson: systematic algebraic probing of embedding spaces with domain-specific knowledge graphs can surface defects that generic benchmarks miss, and the VSA framework provides principled vocabulary for characterizing both what an embedding space can do and where it fails.

All code and data are publicly available.

## References

Bordes, A., Usunier, N., Garcia-Durán, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. *NeurIPS*, 26.

Fong, B., et al. (2025). A Category-Theoretic Foundation for Vector Symbolic Architectures. *arXiv:2501.05368*.

Conneau, A., Kruszewski, G., Lample, G., Barrault, L., & Baroni, M. (2018). What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties. *ACL*.

Ethayarajh, K., Duvenaud, D., & Hirst, G. (2019). Towards understanding linear word analogies. *ACL*.

Gayler, R. W. (2003). Vector Symbolic Architectures answer Jackendoff's challenges for cognitive science. *ICCS/ASCS Joint Conference*.

Gosmann, J., & Eliasmith, C. (2019). Vector-derived transformation binding: An improved binding operation for deep symbol-like processing in neural networks. *Neural Computation*, 31(5), 849–869.

Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. *NAACL*.

Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159.

Kazemi, S. M., & Poole, D. (2018). SimplE embedding for link prediction in knowledge graphs with baseline model comparison. *NeurIPS*.

Kleyko, D., et al. (2023). A survey on hyperdimensional computing: Theory, architecture, and applications. *ACM Computing Surveys*, 55(6), 1–40.


Li, Y., Liu, Y., Deng, G., Zhang, Y., & Song, W. (2024). Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection. *Proceedings of the ACM on Software Engineering*, 1(FSE). https://doi.org/10.1145/3660799

Linzen, T. (2016). Issues in evaluating semantic spaces using word analogies. *RepEval Workshop*.

Liu, Y., Jun, E., Li, Q., & Heer, J. (2019). Latent Space Cartography: Visual Analysis of Vector Space Embeddings. *Computer Graphics Forum*, 38(3), 67–78. (Proc. EuroVis 2019).

Manhaeve, R., Dumančić, S., Kimmig, A., Demeester, T., & De Raedt, L. (2018). DeepProbLog: Neural probabilistic logic programming. *NeurIPS*.

McCoy, R. T., Linzen, T., Dunbar, E., & Smolensky, P. (2019). RNNs implicitly implement tensor-product representations. *ICLR*.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS*.

Nickel, M., Rosasco, L., & Poggio, T. (2016). Holographic embeddings of knowledge graphs. *AAAI*.

Plate, T. A. (2003). *Holographic Reduced Representations*. CSLI Publications.

Rocktäschel, T., & Riedel, S. (2017). End-to-end differentiable proving. *NeurIPS*.

Schlegel, K., et al. (2022). A comparison of vector symbolic architectures. *Artificial Intelligence Review*, 51, 4523–4560.

Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. *Artificial Intelligence*, 46(1-2), 159–216.

Rogers, A., Drozd, A., & Li, B. (2017). The (too many) problems of analogical reasoning with word vectors. *StarSem*.

Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How good is your tokenizer? On the monolingual performance of multilingual language models. *ACL*.

Schluter, N. (2018). The word analogy testing caveat. *NAACL*.

Schuster, M., & Nakajima, K. (2012). Japanese and Korean voice search. *ICASSP*.

Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL*.

Serafini, L., & Garcez, A. d'A. (2016). Logic Tensor Networks: Deep learning and logical reasoning from data and knowledge. *NeSy Workshop*.

Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. *ICLR*.

Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex embeddings for simple link prediction. *ICML*.

Vilnis, L., Li, X., Xiang, S., & McCallum, A. (2018). Probabilistic embedding of knowledge graphs with box lattice measures. *ACL*.

Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). Knowledge graph embedding by translating on hyperplanes. *AAAI*.

Yang, B., Yih, W., He, X., Gao, J., & Deng, L. (2015). Embedding entities and relations for learning and inference in knowledge bases. *ICLR*.
