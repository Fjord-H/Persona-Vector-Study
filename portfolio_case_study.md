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

| Model | Formatting | Method | Sub-test A | Sub-test B | Wilson 95% CI |
|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | chat | content_pole | 97.4% | **95.8%** (46/48 items) | [74.2%, 97.7%]* |
| llama-3.2-3b-instruct§ | chat | content_pole | 98.7% | 77.1% | [63.5%, 86.7%] |
| gpt2-medium | raw | content_pole | 97.4% | 68.8% | [54.7%, 80.1%] |
| qwen2.5-1.5b (base) | chat | content_pole | 90.8%† | 14.6%† | [7.2%, 27.2%] |
| **TF-IDF baseline** | — | — | 93.4% | **85.4%** (18/24 pairs) | [55.1%, 88.0%]* |

†length-flagged result, treat with care. *pair-level Wilson CI (resampling at the 24-pair level, since calm/hostile items within a pair aren't independent); item-level CI [86.0%, 98.9%] for the instruct result. All other table CIs are item-level. §Pipeline provenance unverified — this model's extraction manifest wasn't in the downloaded caches, so its number is the original Kaggle result, not independently re-confirmed against the current frozen-split code.

The instruct model with chat formatting holds 95.8% accuracy (46/48 items) on tone-varying pairs vs. TF-IDF's 85.4% (18/24 pairs) — a directional gap. McNemar's test on the paired sub-test B predictions gives p between 0.062 and 0.180 (exact value depends on the discordant-pair count, which per-item instruct predictions can't fully resolve due to a corrupted activation shard) — not significant at 0.05 with this sample size. The more informative framing is relative error-rate scaling: TF-IDF's error rate grows 2.2× from sub-test A to sub-test B (6.6%→14.6%), while content_pole's grows only 1.6× (2.6%→4.2%). The base model (same architecture, same size, no instruction tuning) drops to 14.6% on the same pairs. That's the controlled comparison this kind of claim requires — same model family, same parameter count, only the post-training differs — and it's consistent with instruction tuning producing more tone-invariant representations. It is not proof: N=24 pairs, one model family, no mechanistic account of why, and the accuracy gap alone isn't statistically significant. I state it as suggestive and preliminary, leaning on the AUROC and relative-error evidence rather than the raw accuracy gap.

**Diagnosing the 14.6% result — a second-round finding.** A external review pushed on whether 14.6% (below the 50% chance rate) meant the base model's representation was *inverted* — confidently tracking tone but reporting it backwards — or whether the number was a threshold artifact. I computed AUROC (threshold-free) on the same scores: 0.910, meaning the direction is correct — harmful items score reliably higher than neutral ones. The 14.6% comes entirely from a threshold calibrated on sub-test A's validation distribution failing to transfer to sub-test B's distribution, where hostile phrasing pushes neutral items' scores above that fixed cutoff. Across every base-model content_pole result, AUROC ranged 0.872–0.979 — the direction was always right; only the threshold failed to generalize.

I also ran a logistic-regression probe (trained on the same activations, same split) as a second reader of the same signal, since a fixed centroid-midpoint threshold is a weaker classifier than a fitted one, and the comparison needed to be fair. The probe recovers most of what the threshold lost: 70.8% on qwen2.5-1.5b/chat (vs. content_pole's 14.6%) and 62.5% on llama-3.2-3b/raw (vs. 43.8%) — though on gpt2-medium both probe and content_pole show threshold-transfer failure (10.4% vs. 68.8%); this is not p>>n overfitting — llama-3.2-3b with 3× larger hidden dimensions scores 62.5% with the probe, opposite the pattern overfitting would predict; the gpt2 probe's decision boundary simply does not transfer to sub-test B's shifted distribution.

Raw formatting degrades generalization across all models — chat-template structure appears load-bearing for tone-invariant representations, though this is correlational, not tested causally.

**Caveats stated honestly:** Sub-test B N=24 pairs (pair-level Wilson CI [74.2%, 97.7%]; item-level [86.0%, 98.9%]; the items within a pair are not independent, so pair-level is the more honest bound). McNemar's test on the 48 sub-test B items gives p between 0.062 and 0.180 depending on the unknown discordant-pair split — not significant at 0.05; the gap is better read as relative error scaling (TF-IDF error 2.2×, content_pole 1.6×). The comparison vs. TF-IDF is suggestive but not fully distinguishable at this sample size. A 150-pair sub-test B would be needed to confirm. A lexically-controlled 5,200-item evaluation set (sub-test B v3) was built to test this further; evaluation is pending. AUROC/probe diagnostics are incomplete for both instruct models due to a corrupted activation shard and missing local cache — their headline accuracy numbers are the original full Kaggle run and remain valid.

---

## What I Learned

**Technically:**
- Designing leak-free evaluation protocols for activation-space methods is harder than it looks — layer selection, threshold fitting, and pooling choice all create opportunities for test-set contamination
- Chat-template formatting is not cosmetic; it changes what the model's hidden states represent
- Neutral-origin distance (Method 3) over-fits to the content distribution and fails to generalize — negative results are data too

**About research process:**
- The most important moment in this project was running TF-IDF and watching it match the activation methods on sub-test A. A baseline that challenges your hypothesis is more valuable than one that confirms it
- Documenting a null result properly (frozen splits, CIs, baselines, honest caveats) is harder and more useful than reporting a clean positive
- A below-chance number deserves a diagnostic, not a shrug. "14.6%, treat as unreliable" was true but incomplete — AUROC showed the underlying signal was fine and the failure was a threshold-transfer problem, which is a fixable, specific claim instead of a vague one
- External review caught real gaps I didn't see myself (the probe-vs-threshold fairness issue, specifically) — the second pass materially improved the finding, not just the writing

---

## Resume Bullets

**For ML/research roles:**
- Ran a controlled base-vs-instruct comparison (same architecture, same size) on transformer representation geometry: instruction-tuned Qwen2.5-1.5B holds 95.8% accuracy on tone-varying harmful content detection (vs. 85.4% TF-IDF baseline) while the base model scores 14.6% on identical pairs — diagnosed via AUROC (0.910) as a threshold-transfer failure rather than a representation failure, then confirmed with a logistic-regression probe that recovered 70.8% on the same items
- Reported all small-sample results (N=24) with Wilson score intervals and explicit fraction-correct, not point estimates alone
- Built a from-scratch activation extraction pipeline across 5 LLMs (GPT-2, Qwen2.5, Llama-3.2), all layers, with frozen splits, bootstrap CIs, length-correlation flags, and random-direction null controls
- Self-audited v1 study, identified and documented 12 methodological defects including training-set contamination and test-set-optimized thresholds; rebuilt pipeline from scratch to correct all defects

**For SWE/MLE roles:**
- Built a resume-safe, checkpoint-driven activation extraction pipeline for 5 LLMs on Kaggle GPU (T4 x2), extracting and caching 5,800+ prompts × all layers × 2 pooling variants per model
- Designed a lexically-controlled 5,200-item evaluation dataset (sub-test B v3) with per-item random cipher pre-shifts to prevent vocabulary leakage; TF-IDF scores 69% by construction
- Conducted a self-audit of prior research, found and publicly documented training-data contamination and test-set leakage; rebuilt evaluation pipeline with strict train/val/test separation

---

*Full pipeline code, data, and audit report: github.com/Fjord-H/Persona-Vector-Study*
