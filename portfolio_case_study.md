# Persona Vector Study: Detecting Harmful Content via Transformer Activations

**Role:** Solo researcher and engineer  
**Duration:** December 2025 – September 2026  
**Stack:** Python, PyTorch, HuggingFace Transformers, scikit-learn, Kaggle GPU (T4 x2)  
**Repo:** github.com/Fjord-H/Persona-Vector-Study

---

## The Question

Can a language model detect harmful content by analyzing its own internal activations — without relying on surface-level keywords or an external classifier?

The motivation is practical: external safety classifiers add latency and compute cost. If a model's hidden states already encode whether a prompt is harmful, you could read that signal directly from the forward pass.

---

## What I Built

A from-scratch evaluation pipeline across 5 transformer models (GPT-2 Medium, Qwen2.5-1.5B base and instruct, Llama-3.2-3B base and instruct), comparing three activation-based detection methods against a TF-IDF bag-of-words baseline.

**Pipeline features:**
- Activation extraction across all layers, two pooling variants (masked-mean, last-token), two formatting variants (raw, chat-templated)
- Frozen train/val/test splits (seed 42, stratified 70/15/15) — no test-set leakage
- Bootstrap confidence intervals on all reported accuracy numbers
- Per-result length-correlation flag to catch confounded pooling variants
- Random-direction null controls to verify methods beat chance
- Resume-safe checkpointing for long Kaggle GPU runs

**Two evaluation sets:**
- Sub-test A (550 items): content varies, tone is neutral — tests whether activations separate harmful from benign content
- Sub-test B (48 items): content is fixed, tone varies between calm and hostile — tests whether the learned representations are tone-invariant

---

## The v1 Problem (and Why Honesty Matters)

The original study (v1) reported 92.5% accuracy. An August 2026 self-audit found the validation set was contaminated: 39 of 40 "new" validation examples appeared verbatim in the training data. The real held-out accuracy was 44–61%, below a TF-IDF baseline.

Rather than quietly fixing it, I documented every defect publicly in `defect_report.md` — 12 issues covering data leakage, test-set-optimized thresholds, incompatible comparison metrics, and asymmetric layer coverage across models. The v2 pipeline was built to correct all of them.

This matters for the portfolio: the ability to find and disclose your own errors is a research skill. The audit is part of the work.

---

## Results

**Sub-test A — content detection:**

Activation methods reach 93–100% accuracy across models. So does TF-IDF at 93.4%. The separation is largely lexical — HarmBench prompts and neutral prompts are distributionally distinct as text, so bag-of-words nearly saturates the task. Activation geometry adds little here.

**Sub-test B — tone invariance (the interesting part):**

| Model | Formatting | Method | Sub-test A | Sub-test B | A→B drop |
|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | chat | content_pole | 97.4% | **95.8%** | **1.5 pp** |
| llama-3.2-3b-instruct | chat | content_pole | 98.7% | 77.1% | 21.6 pp |
| gpt2-medium | raw | content_pole | 97.4% | 68.8% | 28.6 pp |
| qwen2.5-1.5b (base) | chat | content_pole | 90.8%† | 14.6%† | — |
| **TF-IDF baseline** | — | — | 93.4% | **85.4%** | — |

†length-flagged result; treat as unreliable.

The instruct model with chat formatting holds 95.8% accuracy on tone-varying pairs — 10 percentage points above TF-IDF on the same pairs. The base model (same architecture, same size, no RLHF) collapses to 14.6%. That's the controlled comparison this kind of claim requires — same model family, same parameter count, only the post-training differs — and the result is consistent with instruction tuning producing more tone-invariant representations. It is not proof: N=24 pairs, one model family, no mechanistic account of why. I state it as a suggestive, preliminary finding, not a general claim about RLHF.

Raw formatting degrades generalization across all models — the chat template structure (system prompt + user turn) appears load-bearing for building tone-invariant representations.

**Caveats stated honestly:** Sub-test B N=24 pairs. The gap is in the right direction but not statistically distinguishable at this sample size. A 150-pair sub-test B would be needed to confirm. A lexically-controlled 5,200-item evaluation set (sub-test B v3) was built to test this further; evaluation is pending.

---

## What I Learned

**Technically:**
- Designing leak-free evaluation protocols for activation-space methods is harder than it looks — layer selection, threshold fitting, and pooling choice all create opportunities for test-set contamination
- Chat-template formatting is not cosmetic; it changes what the model's hidden states represent
- Neutral-origin distance (Method 3) over-fits to the content distribution and fails to generalize — negative results are data too

**About research process:**
- The most important moment in this project was running TF-IDF and watching it match the activation methods on sub-test A. A baseline that challenges your hypothesis is more valuable than one that confirms it
- Documenting a null result properly (frozen splits, CIs, baselines, honest caveats) is harder and more useful than reporting a clean positive

---

## Resume Bullets

**For ML/research roles:**
- Ran a controlled base-vs-instruct comparison (same architecture, same size) on transformer representation geometry: instruction-tuned Qwen2.5-1.5B holds 95.8% accuracy on tone-varying harmful content detection (vs. 85.4% TF-IDF baseline) while the base model collapses to 14.6% on identical pairs — a result consistent with an RLHF effect, reported with its N=24 sample-size caveat rather than overclaimed
- Built a from-scratch activation extraction pipeline across 5 LLMs (GPT-2, Qwen2.5, Llama-3.2), all layers, with frozen splits, bootstrap CIs, length-correlation flags, and random-direction null controls
- Self-audited v1 study, identified and documented 12 methodological defects including training-set contamination and test-set-optimized thresholds; rebuilt pipeline from scratch to correct all defects

**For SWE/MLE roles:**
- Built a resume-safe, checkpoint-driven activation extraction pipeline for 5 LLMs on Kaggle GPU (T4 x2), extracting and caching 5,800+ prompts × all layers × 2 pooling variants per model
- Designed a lexically-controlled 5,200-item evaluation dataset (sub-test B v3) with per-item random cipher pre-shifts to prevent vocabulary leakage; TF-IDF scores 69% by construction
- Conducted a self-audit of prior research, found and publicly documented training-data contamination and test-set leakage; rebuilt evaluation pipeline with strict train/val/test separation

---

*Full pipeline code, data, and audit report: github.com/Fjord-H/Persona-Vector-Study*
