# Model Provenance Constitution

## Metadata

| Field | Value |
|-------|-------|
| **Code** | Model Provenance Constitution |
| **Full Name** | Model Provenance: Taxonomy, Definition, and Boundary Specification |
| **Version** | 1.0.0 |
| **Domain** | AI Security / ML Supply Chain / Model Governance |
| **Conservatism** | High — under ambiguity, a pair is considered provenance-independent. |
| **Authority** | This is a condensed public reference. The internal specification of Model Provenance Constitution governs where the two differ. |

### Standards Mappings

| Standard | Code(s) |
|----------|---------|
| NIST AI 100-1 (AI RMF) | Map 1.1, 1.5; Measure 2.5; Manage 1.1 |
| NIST AI 600-1 | Model integrity; supply-chain trust |
| NIST SP 800-218 (SSDF) | PW.4.1 |
| NIST SP 800-53 Rev. 5 | SA-10, SR-3, SR-4 |
| EU AI Act (2024) | Arts. 15–17; Annex IV |
| ISO/IEC 42001:2023 | AI management-system lifecycle |
| ISO/IEC 5338:2023 | AI lifecycle provenance and traceability |
| MITRE ATLAS | AML.T0010, T0018, T0044, T0048 |
| OWASP ML Top 10 | ML04, ML06 |
| OWASP LLM Top 10 | LLM03, LLM05 |
| Cisco AI Security and Safety Framework | OB-009; AITech-9.2, AITech-9.3 |
| FS-ISAC MRM | Third-party model validation and lineage |

---

## 1. Purpose and Scope

This document defines what constitutes a provenance relationship between two machine-learning models, and what does not. It is scoped to transformer-based language models.

**In scope:** direct weight derivation, indirect weight derivation via distillation, structural modification with weight inheritance, identity and reformatting, and the set of relationships that appear related but carry no weight lineage.

**Out of scope:** data provenance, behavioral equivalence, authorship attribution, detection methodology, severity or legal-risk scoring, non-transformer architectures.

---

## 2. Definition

**Model provenance** is the verifiable derivation history of a model's trained weights — the chain of initializations, training operations, and mechanical transformations that produced the current parameters from a prior model's parameters.

Two models are **provenance-linked** if and only if one of the following holds:

| # | Condition | Statement |
|---|-----------|-----------|
| C1 | Direct descent | B's training was initialized from A's trained checkpoint. |
| C2 | Indirect descent | B was trained with A as a distillation teacher; B's weights are a function of A's weights. |
| C3 | Mechanical transformation | B was produced from A's weights by a non-training operation (quantization, pruning, merging, format conversion). |
| C4 | Identity | B is a byte-level or numerically equivalent copy of A. |
| C5 | Transitivity | Any chain of C1–C4 is itself a provenance link. |

Weight lineage is **symmetric for detection**: the relationship is reported, not the direction.

The following, taken alone, do **not** establish weight lineage: shared architecture, family name, organization, training data, random seed, hyperparameters, tokenizer, benchmark performance, or methodological inspiration.

---

## 3. Provenance-Linked Relationships

| # | Category | Definition | Canonical example |
|---|----------|------------|-------------------|
| 3.1 | Identity and reformatting | The model is a byte-identical copy or a lossless format conversion of the original weights. | Re-upload; SafeTensors ↔ GGUF conversion |
| 3.2 | Fine-tuning (SFT, RLHF, DPO) | Training resumed from the parent's checkpoint on new data or objectives to specialize behavior. | Llama-2-7B → Llama-2-7B-chat |
| 3.3 | Continued pretraining | Extended pretraining of the parent's checkpoint on additional or domain-specific corpora. | Llama-2-7B → CodeLlama-7B |
| 3.4 | Vocabulary-modified derivation | The parent's weights are adapted to a new or expanded tokenizer vocabulary, with weight inheritance for shared tokens. | RoBERTa → SecureBERT |
| 3.5 | Knowledge distillation | A smaller student model is trained to reproduce the parent's output distributions, transferring learned behavior through a distillation loss. | BERT → DistilBERT; BERT → TinyBERT |
| 3.6 | Structural modification / pruning / scaled transfer | Layers or parameters are removed, rearranged, or selectively transferred from the parent to produce a structurally different model. | Phi-1.5 → Phi-2 |
| 3.7 | Quantization and compression | The parent's weights are converted to a lower-precision numerical format to reduce model size or inference cost. | FP16 → INT4 (GPTQ, AWQ) |
| 3.8 | Adapter-based derivation (LoRA, QLoRA) | Lightweight adapter parameters trained on top of the parent are merged back into the base weights. | Any LoRA merge of a base model |
| 3.9 | Model merging and interpolation | Weights from two or more models (at least one being the parent) are mathematically combined into a single set of parameters. | SLERP / TIES / DARE / Frankenmerges |

---

## 4. Provenance-Independent Relationships

| # | Category | Why it is not provenance |
|---|----------|--------------------------|
| 4.1 | Independent reproduction | Same architecture and tokenizer, trained from scratch (e.g., Llama-2 vs. Open LLaMA). |
| 4.2 | Same family, different size | Each size independently trained from random initialization (e.g., Llama-2-7B vs. Llama-2-13B). |
| 4.3 | Same family, different training corpus | Shared name root, separate from-scratch training (e.g., T5 vs. mT5). |
| 4.4 | Same architecture, independent training runs | Shared seed does not constitute shared weights (e.g., Pythia vs. Pythia-deduped). |
| 4.5 | Architectural convergence | Independent designs adopting the same community best practice. |
| 4.6 | Similar dimensions, different mechanism | Matching dimensions, different internal computation (e.g., BERT vs. ALBERT). |
| 4.7 | Shared vocabulary, independent weights | A tokenizer is a tool, not a weight. |
| 4.8 | Same objective, different everything else | Sharing a loss function does not link weights. |

---

## 5. Decision Logic

```
For a pair (A, B):

  1. Copy or format conversion?         ─YES─► linked (§3.1)
                │ NO
                ▼
  2. Initialized from A's checkpoint?   ─YES─► linked (§§3.2–3.4)
                │ NO
                ▼
  3. Distilled from A?                  ─YES─► linked (§3.5)
                │ NO
                ▼
  4. Mechanical transform of A?         ─YES─► linked (§§3.6–3.9)
                │ NO
                ▼
  5. Any chain of C1–C4 connecting A, B? ─YES─► linked (transitivity)
                │ NO
                ▼
                              provenance-independent
```

**Evidence standard.** A label requires (a) official documentation naming the parent and mechanism, (b) checkpoint verification (hashes, layer-by-layer comparison, reproducible derivation scripts), or (c) peer-reviewed third-party analysis. Architecture similarity, naming conventions, and folklore are not sufficient.

---

## 6. Conservatism Stance

**HIGH.** Under uncertainty, pairs are labeled provenance-independent. Over-inclusive definitions dilute the meaning of "linked" and erode trust; under-inclusive definitions are acceptable in defense-in-depth contexts where licensing audits and forensic analysis catch missed cases.

Application: shared family name, organization, architecture, or statistical weight similarity are insufficient on their own. Explicit author denials are respected unless contradicted by checkpoint evidence.

---

## 7. Boundary Conditions

The boundary is **causal weight dependence**: did one model's trained weights play a causal role in producing the other's?

| Provenance-linked | Not provenance-linked |
|-------------------|----------------------|
| BERT → DistilBERT (distillation) | BERT vs. RoBERTa (convergent design) |
| Llama-2-7B → CodeLlama-7B (continued pretraining) | Llama-2-7B vs. Llama-2-13B (independent runs) |
| Base → LoRA-merged variant | Pythia vs. Pythia-deduped |

**Provenance ≠ behavior.** Identical outputs do not imply shared weights. **Provenance ≠ licensing.** The factual derivation question is separate from the legal obligation question. **Multi-parent.** A merged model is linked to all contributing parents.

---

## 8. Representative Examples

**Linked**

| A | B | Mechanism |
|---|---|-----------|
| Llama-2-7B | Llama-2-7B-chat | Fine-tuning (RLHF) |
| BERT-base | DistilBERT | Distillation |
| Mistral-7B | Zephyr-7B-beta | SFT + DPO |
| Llama-2-7B | CodeLlama-7B | Continued pretraining + vocab extension |
| RoBERTa-base | SecureBERT | Continued pretraining + vocab rebuild |
| Phi-1.5 | Phi-2 | Scaled knowledge transfer |

**Independent**

| A | B | Reason |
|---|---|--------|
| Llama-2-7B | Open LLaMA 7B | Independent reproduction |
| Llama-2-7B | Llama-2-13B | Same family, different size |
| T5-small | mT5-small | Same family, different training |
| BERT-base | RoBERTa-base | Architectural convergence |
| BERT-base | ALBERT-base-v2 | Similar dimensions, different mechanism |
| Pythia-1.4B | Pythia-1.4B-deduped | Same architecture, independent runs |

---

## 9. Threat Model

Upstream model components are supply-chain dependencies (Cisco AI Security and Safety Framework: **OB-009** Supply Chain Compromise; **AITech-9.3** Dependency / Plugin Compromise). Provenance detection is the technical control that makes the dependency auditable.

| Adversary goal | Representative technique | Fundamental limit |
|----------------|-------------------------|-------------------|
| **Concealment** of a derivation link | Metadata rewriting, tokenizer swap, aggressive fine-tune, self-distillation | Fully evading all weight-level signals is as expensive as training from scratch; cheaper techniques leave residue (Cisco AITech-9.2 *Detection Evasion*). |
| **Fabrication** of a derivation link | Copy target metadata, adopt target tokenizer, brief fine-tune from target | Trivial at the metadata surface; defeated by weight-level analysis, which cannot be faked without actual weight transfer. |
| **Laundering** the specific parent | Re-train through an unrelated permissively-licensed model | Residual traces of the true parent persist and are recoverable by weight-level forensics. |

**Robust provenance detection must rely on weight-level signals.** Metadata-only systems are trivially defeated.

---

## 10. Known Gaps

This constitution aims for comprehensive coverage but acknowledges areas where the definition of provenance is contested, under active debate, or outpaced by emerging techniques. Each gap below represents a genuinely open question rather than a fixed position.

### 10.1 Inherent Ambiguities

1. **Synthetic-data training.** If Model A generates a dataset and Model B is trained on that dataset from random initialization, is B provenance-linked to A? Under the current definition: **no** — a dataset is data, not weights, and B's weights are not a function of A's weights in the sense of §2. However, B's *capabilities* are clearly shaped by A's *capabilities*, and the philosophical tension between capability derivation and weight derivation is real. The boundary between formal distillation (§3.5), which uses a distillation loss referencing A's outputs, and pure synthetic-data training, which does not, is increasingly fine-grained. As synthetic-data pipelines become more sophisticated — self-instruct, constitutional-AI–style bootstrapping, rejection-sampling fine-tunes whose teacher is another model — this gap may require revision.

2. **Strong modification.** Provenance is binary under this constitution; modification is continuous. A model fine-tuned for 100 steps is clearly derived. A model fine-tuned over a very large token budget on entirely unrelated data may have essentially no residual similarity to the parent. The provenance relationship is *factually* present in both cases; the *detectable* signal ranges from trivial to near-zero. This constitution treats provenance as a factual property of the derivation chain, independent of signal strength, but this commitment creates a tension between what is *true* about a pair and what is *verifiable* about a pair. Practical governance may need to distinguish these cases explicitly.

3. **Multi-hop transitivity.** Condition C5 (§2.2) makes transitivity formal: if A → B → C → D each satisfy C1–C4, then A is provenance-linked to D. In practice, however, each intermediate step introduces modification, and the detectable signal of A's contribution to D may be negligible after enough hops. Transitivity is a logical property of the relationship, not a detection guarantee. The constitution does not currently distinguish "one-hop" from "multi-hop" provenance, but governance frameworks that attach obligations to derivation depth may need to.

4. **Indirect teacher chains.** If B is distilled from an ensemble that includes A among several teachers, B is provenance-linked to all teachers (§3.5). The strength of each individual lineage signal is diluted, but each causal relationship is real. The constitution currently treats all ensemble members as equal provenance ancestors; whether proportional attribution is warranted (e.g., for license allocation) is an open question.

### 10.2 Evolving Techniques

5. **Multi-parent merging at scale.** Weight-merging methods (SLERP, TIES, DARE, task arithmetic, Frankenmerges) convert the provenance graph from a tree into a directed acyclic graph. A merged model is provenance-linked to *every* contributing parent, and community practice now regularly produces models with five, ten, or more parents. Per-parent attribution, the treatment of near-zero merge coefficients, and the composability of license obligations across many simultaneous parents are not yet standardized.

6. **Mixture of Experts.** MoE architectures contain expert sub-networks that may be independently trained, shared across models, or recycled from prior releases. A single MoE model can carry distinct provenance for different experts: some derived from an upstream base, others independently trained, others shared across multiple downstream models. Component-level provenance analysis — attributing each expert separately — is required and is not covered in depth by the current taxonomy.

7. **Neural architecture search.** Models produced by NAS may inherit weight initializations from a search population rather than from a single named parent. The provenance structure is evolutionary: a population of candidates, selective recombination, and possible weight reuse across generations. Mapping NAS pipelines onto a parent–child model is an open modelling question.

8. **Federated learning.** In federated training, weights are a function of contributions from many local models. The resulting global model is derived from all participants, but no single participant is a "parent" in the sense used here. Provenance in federated settings is inherently multi-source, partial, and privacy-constrained. The current constitution does not address this setting directly.

9. **Multimodal composition.** Models that combine language, vision, audio, or other modalities frequently inherit different components from different parents — for example, a language backbone from one foundation model and a vision encoder from another. Whole-model provenance is insufficient; per-modality and per-component provenance must be tracked. Extending the taxonomy to multimodal artifacts is an ongoing area of work.

10. **Sparse-to-dense and dense-to-sparse conversions.** Emerging techniques that convert between sparse and dense representations — extracting dense models from sparse MoE, distilling dense models into structured-sparse approximations — create new weight-transformation categories that do not map cleanly onto the current mechanical-transformation subtypes (§3.7, §3.8). Additional taxonomy entries may be required as these techniques mature.

---

## 11. Maintenance

This is a living document. Updates are expected when new derivation mechanisms emerge, when regulatory frameworks issue formal definitions, or when operational experience requires recalibration of the conservatism stance. The internal specification of Model Provenance Constitution provides the full treatment; this public summary is the external reference.
