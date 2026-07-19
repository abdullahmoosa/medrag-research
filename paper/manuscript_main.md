# A Novelty-Focused, Resource-Realistic Study of Medical RAG Optimization on MedQA USMLE

## Abstract
Most medical RAG papers emphasize component proposals, but fewer isolate which end-to-end choices reliably improve benchmark accuracy under realistic local compute constraints. We present a novelty-focused empirical study of **configuration-level innovation**: a reproducible framework for identifying high-impact RAG decisions from completed experimental runs only. Using MedQA USMLE test (n=1273), local RTX 3090 execution, and strict completion filtering (`total==1273`), we compare no-RAG and RAG variants across retrieval mode, reranking, query reformulation, coarse retrieval, prompting mode, index family, and LLM backbone. The best completed configuration (MedEmbed index + dense retrieval + reranker + reformulation + zero-shot) reaches 0.6049 accuracy (770/1273). It outperforms the best zero-shot no-RAG baseline (0.5593) by +0.0456 absolute (95% CI: +0.0165 to +0.0754; McNemar p=0.0035). Across matched ablations, reranking (+0.0135 mean) and reformulation (+0.0091 mean) are consistently beneficial, while hybrid retrieval underperforms dense retrieval in every matched comparison. The primary novelty is not a new model component; it is a statistically grounded, decision-complete optimization methodology that yields reproducible gains over alternative pipelines.

## 1. Introduction
Medical QA performance claims are often difficult to translate into deployment decisions. In practice, teams must decide among many interacting RAG options (retrieval mode, reformulation, reranking, prompt style, index family) under strict hardware limits. Existing reporting often under-specifies which choices are robust vs incidental.

This paper addresses that gap by framing novelty as **high-confidence configuration discovery** rather than architectural invention. We ask: with fixed saved experiments and no new training/inference, which pipeline choices demonstrably improve MedQA USMLE performance in a local GPU setting?

### 1.1 Novelty Claims
Our novelty is methodological and operational:
1. **Decision-complete RAG optimization protocol** for medical QA, with strict run-quality filtering and deterministic artifact generation.
2. **Paired statistical comparability framework** (McNemar + bootstrap CI) for head-to-head pipeline claims.
3. **Practical contribution**: a validated recipe that improves accuracy over alternative no-RAG and weaker RAG configurations without additional model training.

## 2. Experimental Design for Fair Comparability
### 2.1 Dataset, Compute, and Inclusion Rule
- Dataset: MedQA USMLE test set.
- Runtime context: all runs executed locally on NVIDIA RTX 3090.
- Inclusion rule: only completed runs with `total=1273`.
- Included: 53 completed runs (50 RAG, 3 NO_RAG); 2 incomplete runs excluded.

### 2.2 Compared Alternatives ("Others")
Within the same benchmark and artifact framework, we compare against:
1. **No-RAG baselines** (`zero_shot`, `cot`).
2. **RAG retrieval variants** (`dense` vs `hybrid`).
3. **Query strategy variants** (`reformulation` vs `no_reformulation`).
4. **Reranking variants** (`reranker_on` vs `reranker_off`).
5. **Coarse retrieval variants** (`coarse_off`, `coarse_k20`, `coarse_k30`).
6. **Index families**: `index_1` (BGE, `BAAI/bge-large-en-v1.5`) vs `medembed`.
7. **LLM backbones**: `llama3_med42_8b`, `gemma3`.

This comparison space is the basis for all “comparable to others” claims in this paper.

### 2.3 Statistical Protocol
For core head-to-head comparisons, we use:
1. Paired accuracy deltas.
2. 95% paired bootstrap CIs (3000 samples, seed=42).
3. McNemar exact test on paired correctness transitions.

## 3. Results: Improvement Over Alternatives
### 3.1 Best Configuration vs All Completed Alternatives
The top completed run among all 53 included runs is:
- `RAG | medembed | dense | coarse_k20 | reranker_on | reformulation | llama3_med42_8b | zero_shot`
- Accuracy: **0.604870** (770/1273).

This run is ranked #1 in `table2_top10_completed_configurations.csv`.

### 3.2 Zero-Shot RAG vs Zero-Shot No-RAG (Primary Comparative Claim)
- Best zero-shot RAG: 0.604870
- Best zero-shot no-RAG: 0.559309
- Absolute gain: **+0.045562**
- Relative gain: **+8.15%**
- 95% CI: [+0.016496, +0.075412]
- McNemar p=0.003487

This is the strongest statistically supported improvement in the study.

### 3.3 CoT RAG vs CoT No-RAG
- Best CoT RAG: 0.603299
- Best CoT no-RAG: 0.597015
- Absolute gain: +0.006284
- Relative gain: +1.05%
- 95% CI: [-0.019639, +0.033778]
- McNemar p=0.684176

CoT RAG is directionally better but not statistically significant in this run set.

### 3.4 MedEmbed vs BGE Index Families (Best-vs-Best)
- Best MedEmbed-family run: 0.604870
- Best BGE-family (`index_1`) run: 0.603299
- Delta: +0.001571 (MedEmbed higher)
- 95% CI: [-0.021210, +0.024352]
- p=0.945006

Interpretation: in this dataset/scope, **configuration choices** (reranking, reformulation, retrieval mode) matter more than index-family difference at the top end.

## 4. Why Performance Improved
### 4.1 Factor Effects Across Matched Pairs
From `ablation_effects.csv`:
1. `reranker_on` vs `reranker_off`: mean delta **+0.01350** (14 positive, 2 negative pairs).
2. `reformulation` vs `no_reformulation`: mean delta **+0.00908** (14 positive, 2 negative).
3. `hybrid` vs `dense`: mean delta **-0.01846** (0 positive, 4 negative).
4. `coarse_k20` vs `coarse_off`: near-neutral mean delta (+0.00032).

These aggregate effects show that gains are primarily driven by reranking and reformulation, while dense retrieval dominates hybrid retrieval in this run matrix.

### 4.2 Interaction Analysis: Which Technique Helped Where
From `technique_effect_by_dimension.csv`, the improvements are not uniform; they interact with index family and LLM backbone.

1. **Reranker effect is positive across both index families**
- On BGE (`index_1`): mean +0.01208
- On MedEmbed: mean +0.01493

2. **Reformulation is substantially stronger on MedEmbed slices**
- On BGE (`index_1`): mean +0.00442
- On MedEmbed: mean +0.01375
- Interpretation: reformulation quality appears to convert to retrieval gains more effectively in the MedEmbed branch.

3. **Backbone-specific sensitivity**
- Reranker effect:
  - `gemma3`: mean +0.02347
  - `llama3_med42_8b`: mean +0.00353
- Reformulation effect:
  - `gemma3`: mean +0.00462
  - `llama3_med42_8b`: mean +0.01355
- Interpretation: reranker contributes larger gains for `gemma3`, while reformulation contributes larger gains for `llama3_med42_8b`.

4. **Hybrid retrieval never wins in matched pairs**
- BGE slices: mean hybrid-minus-dense = -0.01807
- MedEmbed slices: mean hybrid-minus-dense = -0.01964
- No positive hybrid-minus-dense matched pair was observed.

### 4.3 Strongest and Weakest Slice-Level Effects
From `technique_highlights.json` and `technique_pair_details.csv`:

1. **Largest positive reranker gain**
- +0.02985 on `medembed | dense | coarse_off | no_reformulation | zero_shot | gemma3`

2. **Largest positive reformulation gain**
- +0.01964 on `medembed | dense | coarse_k20 | reranker_on | zero_shot | llama3_med42_8b`

3. **Largest hybrid failure vs dense**
- -0.03771 on `index_1 | coarse_off | reranker_off | no_reformulation | zero_shot | llama3_med42_8b`

4. **Coarse retrieval effect stays small**
- Strongest positive and negative swings are symmetric at about ±0.00471, confirming coarse retrieval is secondary relative to reranking/reformulation/retrieval-mode choice.

### 4.4 Error Transition and Question-Type Evidence
For the primary zero-shot comparison (`error_transition_summary.csv`):
1. wrong->correct = 220
2. correct->wrong = 162
3. net favorable transitions = +58

By question-type categories from `error_analysis.csv`, wrong->correct transitions were most frequent in:
1. `other`: 113
2. `negation`: 58
3. `treatment`: 25
4. `diagnosis`: 14
5. `mechanism`: 10

This indicates the accuracy gain is distributed across multiple question categories rather than concentrated in a single narrow subtype.

## 5. Comparative Positioning
This work is comparable to “other approaches” at three levels:
1. **Against no-RAG**: clear, significant zero-shot gain.
2. **Against alternative RAG pipelines** in the same study matrix: top recipe consistently ranks above alternatives.
3. **Across index families and prompt modes**: improvements are primarily driven by reranking + reformulation + dense retrieval, not by a single family label alone.

The novelty is therefore a robust optimization framework that identifies and validates this recipe under practical constraints.

## 6. Practical Recipe
For MedQA USMLE under local RTX 3090 constraints, the highest-performing and most defensible recipe in this artifact set is:
1. Use RAG (not no-RAG).
2. Use dense retrieval (not hybrid in this setup).
3. Enable reranker.
4. Enable query reformulation.
5. Prefer zero-shot prompt mode for the strongest measured gain vs no-RAG.

## 7. Limitations
1. Single benchmark dataset (MedQA USMLE test) in manuscript scope.
2. No external benchmark extension (e.g., MedMCQA) in this paper version.
3. Some prediction files store non-unique `example_id="unknown"`; paired analyses are aligned by deterministic row order.
4. Claims are benchmark-performance claims, not clinical deployment efficacy claims.

## 8. Reproducibility and Traceability
All claims are generated from artifact pipelines in this repository:
- `paper/repro/scripts/generate_paper_artifacts.py`
- `paper/repro/scripts/validate_paper_artifacts.py`
- Tables: `paper/artifacts/tables/`
- Figures: `paper/artifacts/figures/`
- Config: `paper/repro/config/analysis_config.json`

## 9. Ethical and Clinical Use Note
This study evaluates QA benchmark performance and configuration sensitivity only. It does not establish clinical safety or utility for patient care without prospective clinical validation and oversight.
