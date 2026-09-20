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

**Sub-test B — distribution-shift robustness (the interesting part):**

| Model | Formatting | Method | Sub-test A | Sub-test B | Wilson 95% CI |
|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | chat | content_pole | 97.4% | **95.8%** (46/48 items) | [74.2%, 97.7%]* |
| gpt2-medium | raw | content_pole | 97.4% | 68.8% | [54.7%, 80.1%] |
| qwen2.5-1.5b (base) | chat | content_pole | 90.8%† | 14.6%† | [7.2%, 27.2%] |
| **TF-IDF baseline** | — | — | 93.4% | **85.4%** (18/24 pairs) | [55.1%, 88.0%]* |

†length-flagged result, treat with care. *pair-level Wilson CI (24-pair level; calm/hostile items within a pair are not independent); item-level CI [86.0%, 98.9%] for instruct result. llama-3.2-3b-instruct (77.1%) excluded from primary table — extraction manifest not in local caches, provenance unverified.

The headline needs a narrower statement than "activations beat TF-IDF." Threshold-free comparison (AUROC): TF-IDF 0.920 on the 54-item combined arm; base-model content_pole range 0.872–0.979 — comparable discrimination. Because AUROC is comparable, the accuracy gap (95.8% vs. 85.4%, 2/48 vs. 7/48 errors) is best explained by which fixed threshold happened to transfer better to the shifted distribution, not by activations carrying more usable signal — and it's not significant either way (McNemar p = 0.062–0.180; pair-level CIs overlap). Error-rate scaling: TF-IDF's error rate grows 2.2× (6.6%→14.6%), content_pole's 1.6× (2.6%→4.2%) — a real but unresolved-in-mechanism difference, most likely calibration rather than representation.

**What sub-test B actually tests:** A fourth-round investigation found that sub-test B neutral items differ from sub-test A neutral in *register*, independent of the calm/hostile tone split. Topics match exactly (same 6 buckets). But sub-test A neutral is 73% imperative-start, mean 66 chars, 0% polite markers. Sub-test B calm is 0% imperative, 92% question-start, 42% polite markers, mean 122 chars. Sub-test B hostile is 100% personal pronouns, 79% demand-register, mean 112 chars. TF-IDF's 4 calm / 3 hostile errors confirm the degradation is register-driven, not tone-driven. The "tone-invariance" framing is not supported by the data; the honest claim is "less degradation under register/distribution shift."

**Diagnosing the 14.6% result — a second-round finding.** A external review pushed on whether 14.6% (below the 50% chance rate) meant the base model's representation was *inverted* — confidently tracking tone but reporting it backwards — or whether the number was a threshold artifact. I computed AUROC (threshold-free) on the same scores: 0.910, meaning the direction is correct — harmful items score reliably higher than neutral ones. The 14.6% comes entirely from a threshold calibrated on sub-test A's validation distribution failing to transfer to sub-test B's distribution, where hostile phrasing pushes neutral items' scores above that fixed cutoff. Across every base-model content_pole result, AUROC ranged 0.872–0.979 — the direction was always right; only the threshold failed to generalize.

I also ran a logistic-regression probe (trained on the same activations, same split). A fourth-round refitting pass standardized features (StandardScaler, fit on train only) and tuned C on the validation split (swept [0.001, 0.01, 0.1, 1, 10, 100]). Results: qwen2.5-1.5b/chat — 60.4% accuracy (AUROC 0.965, vs. 70.8% untuned); llama-3.2-3b/raw — 56.2% (AUROC 0.990, vs. 62.5%); gpt2-medium/raw — 50.0% (AUROC 0.851, vs. 10.4% untuned, a +39.6 pp improvement). The gpt2 result improved dramatically: the outlier activation dimensions were being penalized unevenly by untuned L2 — the original 10.4% was a probe implementation failure, not evidence that gpt2's representations are bad. The AUROC values (0.851–0.990) are the cleaner summary: the signal is present across all three models; threshold-transfer failure persists regardless of how well the probe is fit.

Raw formatting degrades generalization across all models — chat-template structure appears load-bearing for robustness under distribution shift, though this is correlational, not tested causally.

**Caveats stated honestly:** Sub-test B N=24 pairs (pair-level Wilson CI [74.2%, 97.7%], item-level [86.0%, 98.9%]; the items within a pair are not independent). McNemar p = 0.062–0.180 (not significant at 0.05); pair-level CIs overlap. A follow-up study would need roughly 4–6× the current pair count to resolve the accuracy gap statistically. The "tone-invariance" framing is replaced by "register/distribution-shift robustness" after the fourth-round investigation (see above). AUROC/probe diagnostics for qwen2.5-1.5b-instruct are incomplete due to a corrupted activation shard; its 95.8% figure is from the original full Kaggle run and remains valid. The pipeline does not persist per-item predictions, which is why McNemar gives a range rather than an exact p-value; this is flagged as a process gap for future runs.

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
- External review caught real gaps I didn't see myself (the probe-vs-threshold fairness issue, the register-shift in sub-test B, the unstandardized probe dimensions) — four rounds of review progressively narrowed and clarified the claim rather than expanding it, and the narrower claim is more defensible
- Following where data leads when it contradicts your framing: the "tone-invariance" story turned out to test register generalization, not purely tone; the honest finding is less headline-grabbing but supported by the evidence

---

## Resume Bullets

**For ML/research roles:**
- Diagnosed an apparent activations-vs-baseline gap (95.8% vs. 85.4% accuracy) down to its actual mechanism using AUROC as a threshold-free comparison: found discrimination ability was statistically comparable (0.920 vs. 0.872–0.979) between methods, correctly attributing the accuracy difference to threshold-calibration transfer rather than overclaiming a representational advantage
- Ran four rounds of self-directed external review on a negative result, each round narrowing the claim to what the data actually supported — including reversing an initial "tone-invariance" framing after diagnosing that a test set varied register and length alongside tone
- Reported all small-sample results (N=24 pairs) with Wilson score intervals, McNemar significance testing, and explicit fraction-correct rather than point estimates alone
- Built a from-scratch activation extraction pipeline across 5 LLMs (GPT-2, Qwen2.5, Llama-3.2), all layers, with frozen splits, bootstrap CIs, length-correlation flags, and random-direction null controls
- Self-audited v1 study, identified and documented 12 methodological defects including training-set contamination and test-set-optimized thresholds; rebuilt pipeline from scratch to correct all defects

**For SWE/MLE roles:**
- Built a resume-safe, checkpoint-driven activation extraction pipeline for 5 LLMs on Kaggle GPU (T4 x2), extracting and caching 5,800+ prompts × all layers × 2 pooling variants per model
- Designed a lexically-controlled 5,200-item evaluation dataset (sub-test B v3) with per-item random cipher pre-shifts to prevent vocabulary leakage; TF-IDF scores 69% by construction
- Conducted a self-audit of prior research, found and publicly documented training-data contamination and test-set leakage; rebuilt evaluation pipeline with strict train/val/test separation

---

*Full pipeline code, data, and audit report: github.com/Fjord-H/Persona-Vector-Study*
