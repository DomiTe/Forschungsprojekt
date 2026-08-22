# Reviewer Feedback — Data Availability Triage

Read-only investigation. No file under `results/` was modified. `paper/neurips_2026.tex` was
not edited. No training or full evaluation sweep was executed.

**Sources located and used:**
- Prof 2's feedback: `~/Downloads/FSP_B_AXIOM_WORKSHOP_2026_TE.pdf` (6 pages, all annotations
  authored by user "tatiana", timestamps 2026-08-21 07:40–08:59). Two copies of this PDF exist
  in Downloads (`FSP_B_AXIOM_WORKSHOP_2026_TE.pdf` and `...(1).pdf`); they are byte-identical
  (same md5), so there is only one actual annotation set. All Prof 2 citations below give the
  exact page and the paper sentence(s) the sticky-note/highlight is geometrically anchored to,
  recovered via PDF coordinate mapping (`pdfminer` line boxes matched against each annotation's
  `/Rect`), not guesswork.
- Prof 1's feedback: **no mail file was found on disk.** I searched the repo, `~/Downloads`,
  `~/Documents`, `~/Desktop` (recursively) and common mail-export locations (`.eml`, `.msg`,
  Thunderbird profiles) and found nothing matching. Prof 1's items below are therefore sourced
  from this task's own brief (which paraphrases the mail operationally: exact formulas, CSV
  names, and column names were specified precisely enough to execute Part 2 without the
  original text). **If exact wording matters for the resubmission email, the original mail
  still needs to be pulled up separately** — I cannot cite a file/line for it the way I can for
  Prof 2's PDF.
- Canonical data: two runs, confirmed as the ones actually cited in the paper by matching the
  Discussion's `13161.8 / 1030.9 / 97.8` trace values byte-for-byte against
  `trace_reconciliation_ledger.csv`:
  - `results/20260816_230437_38678/csv/` — CIFAR10
  - `results/20260816_083054_38677/csv/` — IMAGENET100
- New derived files from Part 2 all written to `results/review_response/csv/` (new directory,
  nothing existing overwritten).

---

## Summary

| Category | Count |
|---|---|
| NO-DATA | 12 |
| COMPUTABLE-NOW | 8 |
| NEEDS-NEW-RUN | 1 |
| LITERATURE-CHECK | 1 |
| NEEDS-CLARIFICATION | 2 |

**Critical path: Item 3.1 (accuracy → loss).** Verdict below is **NEEDS-NEW-RUN**, not free —
loss is computed nowhere in the isolated per-layer eval path. Estimated cost is **~70–150
minutes of GPU time** (not hours of author effort), so it is plausibly still in scope before the
deadline, but nothing about it is a same-session fix, and it requires a small code change before
any rerun. Everything that would *use* an isolated-loss damage metric (a reworked Table 1/2, a
reworked §3.2/§4) should wait on an explicit decision to spend that time, made after reading this
item.

Everything in Part 2 (8 items) is done — all output CSVs exist in `results/review_response/csv/`
now and the headline numbers are quoted inline below with their provenance.

**Value-add beyond the original brief:** the PDF annotation pass surfaced several Prof 2 comments
not enumerated in the task brief (an abstract-framing note, a table-wording tension note, a
self-answered clarification thread, and a Conclusion remark) — these are included below under
"Additional findings" so the text-editing pass has the complete annotation set, not just the
pre-triaged subset.

---

## Part 1 — No data needed (Prof 1, mail)

Unblocked, ready for the text pass, nothing computed:

- Title change ("Back to Basics" or equivalent reframing).
- Cut/keep the random-matrix-theory discussion in Related Work (page budget).
- Cut/keep the KFAC-style approximation discussion in Limitations (page budget).
- Mechanism explanation for why raw trace outperforms the HAWQ product — already drafted in
  Discussion (`paper/neurips_2026.tex:129`); needs sharpening, not new data.

---

## Prof 1 (mail) — items requiring computation or investigation

### CI-1 — Bootstrap confidence intervals for Spearman ρ
Source: Prof 1, mail (PS)
Category: **COMPUTABLE-NOW**
Finding: Computed, B=2000, seed=42, resampling layers with replacement per (model, dataset),
PTQ, from the per-layer `(predictor_value, abs_weight_damage_pts)` pairs. Headline (`raw_trh` /
$S_{raw}$) CIs:

| model | dataset | n | ρ | 95% CI |
|---|---|---|---|---|
| cnn | CIFAR10 | 6 | 0.928 | [0.515, 1.000] |
| cnn | IMAGENET100 | 6 | 0.600 | [−0.800, 1.000] |
| resnet18 | CIFAR10 | 21 | 0.292 | [−0.179, 0.648] |
| resnet18 | IMAGENET100 | 21 | 0.121 | [−0.411, 0.589] |
| resnet50 | CIFAR10 | 54 | 0.205 | [−0.056, 0.449] |
| resnet50 | IMAGENET100 | 54 | 0.299 | [0.010, 0.548] |

All 18 predictor×combo rows (incl. $S_{pert}$, $S_{hawq}$) are in the output file. Note how wide
these are at n=6 (cnn) — the CI, not the point estimate, is the honest way to show the CNN's
apparent "0.93 correlation" is barely distinguishable from a much weaker one at this sample size,
which is exactly Prof 1's point about preferring CIs to p-values here.
Source file(s): `results/20260816_230437_38678/csv/weight_ablation_canonical_v2.csv` and
`results/20260816_083054_38677/csv/weight_ablation_canonical_v2.csv` (columns
`hessian_trace_fused`, `delta_w_sq`, `trh_times_dwsq`, `abs_weight_damage_pts`) →
`results/review_response/csv/bootstrap_ci_spearman.csv`
Recommendation: Replace or supplement Table 2's p-values with these CIs; the width at n=6 is
itself worth a sentence in the text.

### RANK-1 — Normalized (percentile) rank + normalized top-k
Source: Prof 1, mail (PS); reinforced independently by Prof 2, PDF p3 (see B-6 below)
Category: **COMPUTABLE-NOW**
Finding: Computed `rank_pct = true_top_layer_rank / n_layers` and a normalized top-⌈n×0.1⌉
overlap alongside the fixed top-5, per (model, dataset, predictor), PTQ. The fixed-top-5 metric
is visibly generous for the CNN (n=6): e.g. `raw_trh` on cnn/CIFAR10 shows top-5 overlap 5/5 but
normalized top-1 overlap 0/1 (top_k_norm=1 for n=6) — the "5/5" is nearly the whole 6-layer
network, not a meaningful top-k hit. ResNet-18 (top_k_norm=3, i.e. top-14%) and ResNet-50
(top_k_norm=6, i.e. top-11%) are less distorted but still worth reporting alongside fixed-5.
Source file(s): same two `weight_ablation_canonical_v2.csv` files →
`results/review_response/csv/normalized_ranks.csv`
Recommendation: Report both fixed-top-5 and normalized-top-k in Table 1 (or a new column); the
CNN's numbers change the reading substantially, the ResNets' less so.

### NORM-1 — Is $S_{raw}$ already parameter-normalized?
Source: Prof 1, mail (PS)
Category: **COMPUTABLE-NOW**
Finding: **No, $S_{raw}$ as used in Table 1/2 is the raw (un-normalized) Hutchinson trace sum,
not per-parameter.** `_load_canonical_traces()` in `src/analysis/weight_ablation_canonical.py:190`
reads the `trace_raw` column of `canonical_traces.csv` — not `trace_per_param`. A per-parameter
companion column (`trace_per_param = trace_raw / weight.numel()`) is already computed and stored
in the same CSV by `src/analysis/relock_traces.py:229-230,614-636`, but it is not what feeds the
predictor comparison. Using it instead gives a mixed picture, not a clean improvement:

| model | dataset | ρ (raw $S_{raw}$) | ρ (per-param) | top5 overlap (per-param) |
|---|---|---|---|---|
| cnn | CIFAR10 | 0.928 | 0.812 (p=0.050) | 4/5 |
| cnn | IMAGENET100 | 0.600 | **0.943** (p=0.0048) | 4/5 |
| resnet18 | CIFAR10 | 0.292 | 0.457 (p=0.037) | 3/5 |
| resnet18 | IMAGENET100 | 0.121 | 0.219 (p=0.34) | 2/5 |
| resnet50 | CIFAR10 | 0.205 | 0.034 (p=0.81, destroys the signal) | 2/5 |
| resnet50 | IMAGENET100 | 0.299 | 0.369 (p=0.0060) | 2/5 |

Source file(s): `src/analysis/weight_ablation_canonical.py:190`, `src/analysis/relock_traces.py:229-230,382-383,614-636`
(code); `canonical_traces.csv` column `trace_per_param`, variant=`fp32_fused` →
`results/review_response/csv/raw_trace_per_param_comparison.csv`
Recommendation: State plainly in Methods (§3.3) that $S_{raw}$ is unnormalized by design (matches
"HAWQ-style usage" as already stated at `neurips_2026.tex:64`), and optionally add one sentence
noting normalization helps on 4/6 combos and actively hurts on ResNet-50/CIFAR10 — i.e. it is not
a free win, so the paper's choice not to normalize is defensible, not an oversight.

### CONC-1 — Predictor-vs-predictor rank concordance
Source: Prof 1, mail (PS); same ask appears twice from Prof 2 (PDF p3 "Rangkorrelationskoeffizient
könnte auch noch interessant sein" and p4 "Über das gesamte Ranking laufen lassen?") — likely the
same request made independently by both supervisors.
Category: **COMPUTABLE-NOW**
Finding: Pairwise Spearman ρ between $S_{raw}$, $S_{pert}$, $S_{hawq}$ themselves (not vs.
damage), PTQ, all 6 combos (18 pairs total). Two patterns worth naming in the text: (1)
$S_{raw}$ vs $S_{pert}$ is *negative* for cnn and resnet18 on both datasets (ρ = −0.54 to −0.72),
directly consistent with Discussion's claim (`neurips_2026.tex:129`) that curvature and
perturbation are "directly opposed" for conv1-like layers; (2) on ResNet-50, $S_{pert}$ and
$S_{hawq}$ are almost redundant (ρ = 0.83–0.86, p<2e-15 both datasets) — since
$S_{hawq}=S_{raw}\cdot S_{pert}$, this means for ResNet-50 the HAWQ product's ranking is mostly
just $S_{pert}$'s ranking, curvature barely modulates it.
Source file(s): same two `weight_ablation_canonical_v2.csv` files →
`results/review_response/csv/predictor_concordance.csv`
Recommendation: A short sentence in Discussion: "for ResNet-50, $S_{hawq}$ collapses toward
$S_{pert}$ (ρ≈0.84, both datasets), so its failure to beat $S_{raw}$ is effectively $S_{pert}$'s
failure, not a genuine three-way test on that architecture."

### LEDGER-1 — Reconciliation ledger extract for the historical-traces paragraph
Source: Prof 1, mail
Category: **COMPUTABLE-NOW** (extract only, no new computation)
Finding: All 5 ledger rows touching `resnet50_conv1` pulled verbatim:

| quantity | old_value | old_source | canonical_value | ratio | explained_by_knob |
|---|---|---|---|---|---|
| resnet50_conv1_fp32 | 13.87 | prior run notes (unspecified estimator config) | 14.904 | 0.931 | fp32:basis=unfused (residual 2.78%) |
| resnet50_conv1_fp32 | 11.79 | prior run notes, second run | 14.904 | 0.791 | UNRESOLVED |
| resnet50_conv1_ptq | **13161.8** | original `src/main.py` pipeline run, `layerwise_hessian_traces.csv` (PTQ stage) | **97.76** | 134.6× | UNRESOLVED |
| resnet50_conv1_ptq | **1030.9** | a later/fresh run's notes | **97.76** | 10.5× | UNRESOLVED |
| resnet50_conv1_elevation_fp32 | 14.0 | prior "conv1 14x" claim | 11.73 | 1.19× | UNRESOLVED |

The two `_ptq` rows are the ones the paper text cites. Both are marked `UNRESOLVED` — "no knob
[basis, loss reduction, probe count, data source] reproduced this anchor within tolerance." This
is a documented finding (a specific, closed sweep that failed to reconcile), not an open
limitation — the paragraph can say so explicitly rather than reading as an unresolved worry.
Source file(s): `results/20260816_230437_38678/csv/trace_reconciliation_ledger.csv` (unfiltered) →
`results/review_response/csv/reconciliation_ledger_resnet50_conv1.csv` (filtered extract)
Recommendation: Rewrite the Discussion paragraph (`neurips_2026.tex:131`) using this table
directly — state that a systematic sweep (documented in `drift_diagnosis.csv`, not reproduced
here) was run and *closed as unresolved*, not left untried.

---

## Prof 2 (PDF) — items, in page order

### A-1 — Abstract opening: justify relevance
Source: Prof 2, PDF p1, anchored at abstract lines 1–3 ("Hessian-based curvature analyses are a
tool for quantization-sensitivity ranking, typically combining curvature and perturbation
magnitude into a single product... We test this formulation")
Category: **NO-DATA**
Finding: Comment: *"Können wir begründen, warum dies aktuell und wichtig ist? Und hier überprüft
werden soll?"* (Can we justify why this is current/important, and why it should be tested here?)
Not in the original task brief's Part 1 list — new finding from the PDF pass.
Recommendation: Add one framing clause to the abstract opening motivating currency/importance
before "We test this formulation." Argument to write, not a result to compute.

### A-2 — Abstract: add method description
Source: Prof 2, PDF p1, anchored at abstract lines 5–7 ("across three CNN architectures... on
two datasets... Raw curvature...identifies the true single most-damaged layer at rank 1")
Category: **NO-DATA**
Finding: Comment: *"Die Vorgehensweise beschreiben?"* — matches Part 1's pre-listed item exactly.
Recommendation: One clause describing the isolation methodology (weight-only PoT quantization,
per-layer isolation) before jumping to results.

### A-3 — Abstract vs. Table 2 wording tension
Source: Prof 2, PDF p1, highlight spanning abstract lines 12–14 ("...no predictor performs well).
Full-ranking Spearman correlation is weak and mostly non-significant throughout. Robust top-k
identification...is the finding that survives across datasets.")
Category: **NO-DATA**
Finding: Comment: *"In der Tabelle gibt es auch größere Koeffizienten."* (There are also larger
coefficients in the table.) Not in the original task brief. The abstract's blanket "weak...
throughout" sits in tension with Table 2 containing $\rho=0.93$ (CNN/CIFAR10) and $\rho=-0.94$
(CNN/ImageNet100, both $p<0.01$) — the Results text itself is precise ("only 3 of 18...reach
p<0.05", `neurips_2026.tex:123`) but the abstract's phrasing reads as flatly contradicting its
own table at a glance.
Source file(s): Table 2 values already in `neurips_2026.tex:113-119`; no new computation needed.
Recommendation: Soften "weak and mostly non-significant throughout" to something like "weak
overall, with two exceptions" to match the Results section's own precision.

### CLARIFY-1 — "Zwischen was?"
Source: Prof 2, PDF p1, `/Text` annotation at y≈352 (PDF pt), same location as A-3's highlight
(abstract lines 12–14, closing sentences of the abstract).
Category: **NEEDS-CLARIFICATION**
Finding: Verbatim: *"Zwischen was?"* ("Between what?"). Anchored geometrically at exactly the
same lines as A-3 above (the highlight "In der Tabelle gibt es auch größere Koeffizienten." and
this note share the same y-range 340–364pt), so it plausibly continues that thought — but I am
not resolving what specific word/comparison ("correlation," "overlap...never worse than," or
something else) is being questioned. Reported verbatim, not interpreted, per instructions.
Source file(s): `FSP_B_AXIOM_WORKSHOP_2026_TE.pdf`, page 1.
Recommendation: Author confirms directly against the PDF (open p.1, right column, ~40% down)
before this is actioned.

### CITE-1 — Citation completeness audit (Introduction)
Source: Prof 2, PDF p1, `/Text` at Introduction paragraph 1 ("...CIFAR10 using Post-Training-
Quantization, depending on architecture..." / "The Hessian-Aware-Quantization (HAWQ) family
addresses this...")
Category: **COMPUTABLE-NOW**
Finding: Comment: *"Diese Quellen für alle Aussagen?"* Full paragraph-by-paragraph audit done
(12 claims). Almost every external claim is cited; the accuracy-cost numbers and the paper's own
design statements are correctly self-attributed (no citation needed). **One real gap, and it's
more interesting than a missing citation:**

> `neurips_2026.tex:41` currently reads: *"the underlying predictive claim...is rarely tested
> directly against measured damage"* — **uncited**. But the PDF the professors actually
> annotated (`FSP_B_AXIOM_WORKSHOP_2026_TE.pdf`) has *different* text at this exact spot:
> *"...has recently been tested in the LLM setting [Hill, 2026] but its validity under PoT for
> vision CNNs is untested."* `hill_2026` does not exist in `paper/reference.bib` and is not cited
> anywhere in the current `paper/neurips_2026.tex` (single commit `55214a0`, no prior git
> history for this file). **The committed source has silently dropped a citation that was present
> in the version the professors reviewed**, replacing a specific claim with a vaguer, uncited one.
> This is exactly the sentence Prof 2's "Diese Quellen für alle Aussagen?" note is anchored to, and
> it directly overlaps with the LITERATURE-CHECK item below (Prof 2 asking, separately, whether
> HAWQ/HAWQ-V2 themselves tested their metric).

Source file(s): `paper/neurips_2026.tex:37-45`, `paper/reference.bib`, `git log --follow
paper/neurips_2026.tex` → full audit table at
`results/review_response/csv/intro_citation_audit.csv`
Recommendation: Resolve deliberately, not by accident: either restore a citation for the LLM-
setting claim (find/re-verify "Hill, 2026" — this looks like it may have been cut because it
couldn't be verified, in which case leave it cut but say so) or keep the current uncited phrasing
and soften it to something the LITERATURE-CHECK item below actually supports (see LIT-1).

### B-1 — §3.2 weight_damage formula: is it accuracy-based?
Source: Prof 2, PDF p3, `/Text` at §3.2 "Weight-only damage isolation," anchored directly at the
$\mathrm{weight\_damage}(l) = \mathrm{acc}_{fp32} - \mathrm{acc}_{iso}(l)$ definition
(`neurips_2026.tex:61`)
Category: **NEEDS-NEW-RUN** (folds into 3.1 below — same underlying question, asked
independently by Prof 2 in the PDF: *"Weight damage über accuracies formuliert?"*)
Finding: Confirms in her own words what Prof 1's mail separately requests: the damage metric is
currently accuracy-only. See item 3.1 for the full investigation.
Recommendation: See 3.1's recommendation — do not duplicate work, this is the same ask.

### B-2 — "90%→92% and 92%→90%: both 2%?"
Source: Prof 2, mail + PDF p3, `/Text` at the same §3.2 passage ("Because isolated single-layer
quantization can improve accuracy for some layers...")
Category: **COMPUTABLE-NOW**
Finding: Verbatim: *"90%->92% und 92%->90%: Beides 2%?"* — questioning whether a hypothetical
90%→92% improvement and 92%→90% degradation are being treated as symmetric 2-point changes. This
is directly answerable: signed `weight_damage_pts` (not `abs_weight_damage_pts`) does go negative
— isolated single-layer quantization measurably *improves* accuracy in several cases, confirming
this isn't hypothetical:

| stage | model | dataset | # negative rows | magnitude range |
|---|---|---|---|---|
| PTQ | cnn | CIFAR10 | 1 | −0.08 |
| PTQ | resnet18 | CIFAR10 | 11 | −0.03 to −0.26 |
| PTQ | resnet18 | IMAGENET100 | 6 | −0.08 to −0.16 |
| PTQ | resnet50 | CIFAR10 | 33 (of 54!) | −0.01 to **−0.57** |
| PTQ | resnet50 | IMAGENET100 | 19 | −0.02 to −0.24 |

**The single most striking number: the `fc` layer on ResNet-50/CIFAR10 — the very layer Table 1
reports as the "top-damage layer" for that combination (0.57 of 1.6 total points) — has
`weight_damage_pts = −0.57`, i.e. isolating its quantization *raises* accuracy from 81.14% to
81.71%.** It is only "top damage" because Table 1 ranks by $|{\rm weight\_damage}|$
(`abs_weight_damage_pts`); by sign, it is the single best-improving layer in that entire
combination. This is a real, headline-relevant instance of exactly the 90%→92%-style scenario
Prof 2 is asking about — not hypothetical, and worth a sentence in §4.2 or a footnote on Table 1.
QAT rows (bonus, not required by the paper's PTQ-only scope) show negative damage is far more
pervasive there — e.g. all 54 ResNet-50/CIFAR10 layers are individually accuracy-*improving*
under QAT isolation, consistent with Limitations item (3)'s note that QAT damage is "mostly
negative."
Source file(s): both `weight_ablation_canonical_v2.csv` files, column `weight_damage_pts` →
`results/review_response/csv/signed_damage_improving_cases.csv`
Recommendation: Add the `fc`/ResNet-50/CIFAR10 number to §4.2 as a concrete example; it directly
answers Prof 2's question with a real, already-published-elsewhere-in-the-paper data point.

### B-3 — Why these three predictors?
Source: Prof 2, PDF p3, `/Text` at §3.3 predictor definitions ("$S_{raw}(l)=\mathrm{Tr}(H_l)$...")
Category: **NO-DATA**
Finding: *"Begründen, warum diese eine Rolle spielen könnten?"* — matches Part 1's pre-listed
item ("why exactly three predictors, framed as directly derived from HAWQ-V2's formulation").
Recommendation: Argument to write in §3.3, not a computation.

### B-4 — Table 1: "top-5" arbitrary threshold?
Source: Prof 2, PDF p3, `/Text` at Table 1's caption/header row, y≈703–715
Category: **NO-DATA / COMPUTABLE-NOW crossover — resolved by RANK-1 above**
Finding: *"Eher freiwillig gewählt?"* ("Chosen rather arbitrarily?") — not in the original task
brief's item list, but this is Prof 2 independently raising the same concern Prof 1's mail raised
as RANK-1 (normalized top-k). Already computed there.
Recommendation: No separate action — point the text-editing pass to RANK-1's output.

### B-5 — Table 1 notation: "how was this determined?"
Source: Prof 2, PDF p3, `/Text` thread at the ResNet-18/ImageNet100 row (rank=1, all three
top5=1/5)
Category: **NO-DATA** (self-answered in the annotation thread)
Finding: *"Wie wurde dies bestimmt?"* answered by her own follow-up: *"Über die weight_damage-
Formel?"* — she resolves her own question by reference to the §3.2 formula. No action needed
beyond confirming the caption makes this link explicit for other readers.
Recommendation: Make sure Table 1's caption or a footnote explicitly states ranks/overlaps are
computed from `abs_weight_damage_pts` via the §3.2 formula, so this doesn't need re-deriving.

### B-6 — "top 1 place 4 out of 5?" — Table 1 column notation
Source: Prof 2, PDF p3, `/Text` thread at the ResNet-18/ImageNet100 row
Category: **NO-DATA**
Finding: *"Was bedeutet dies: top 1 Platz 4 aus 5?"* / self-follow-up *"Oder top X aus top 5
gefunden?"* — matches Part 1's pre-listed item exactly (Table 1 top-5 column notation clarity).
Recommendation: Clarify the header/caption to state the top-5 column means "count of the true
top-5 damaged layers this predictor's own top-5 recovers," not a rank position.

### B-7 — §4.2 reframe: also architecture-dependent
Source: Prof 2, PDF p3, `/Text` at the §4.2 heading ("Which layer is most damaged is
dataset-dependent")
Category: **NO-DATA**
Finding: *"Aber auch vom Modell"* ("But also on the model") — matches Part 1's pre-listed item
exactly.
Recommendation: Rewrite the section framing sentence; Table 1 already contains the evidence.

### LIT-1 — Did HAWQ/HAWQ-V2 themselves test these metrics?
Source: Prof 2, PDF p3, `/Text` at §3.3, near the $S_{hawq}$ definition and its citation to
Dong et al. 2020
Category: **LITERATURE-CHECK**
Finding: *"Haben sie diese Metriken eingeführt? Und getestet?"* Checked both papers directly
(HAWQ, arXiv:1905.03696; HAWQ-V2, arXiv:1911.03852, via full-text fetch, not abstract-only).
**Both introduce their sensitivity metric but neither directly validates it against measured
per-layer quantization damage:**
- **HAWQ** (Dong et al. 2019): top Hessian eigenvalue ranking is motivated via loss-landscape
  visualizations (Section III-A: "layers with larger Hessian eigenvalue...exhibit larger
  fluctuations in the loss") and validated only via *end-to-end* accuracy after applying the
  resulting mixed-precision scheme (Tables I–IV); the ablation in Section V compares the
  Hessian-based layer *ordering* against a reversed ordering, still end-to-end, never isolating
  one layer to check predicted-vs-measured damage.
- **HAWQ-V2** (Dong et al. 2020): the $\mathrm{Tr}(H)\cdot\|\Delta W\|^2$ product is motivated
  theoretically (Section 2.1, Lemma 1) and used instrumentally to sort layers for a Pareto-
  frontier bit-width search (Section 2.3); Figure 2 shows trace *varies* across blocks (not that
  it predicts damage). No isolated per-layer quantization experiment appears anywhere.

This directly supports the paper's current framing (testing an assumption HAWQ's authors made,
not re-testing something they already validated) — and gives it citable backing that the
"rarely tested" claim currently lacks (see CITE-1 above).

**Caveat:** these findings came from an automated full-text fetch-and-summarize pass over each
paper's arXiv HTML rendering, not a manual read of the PDF. The section numbers and quotes above
should be spot-checked against the actual papers before being asserted in the resubmission
response, but they are specific enough to act as a strong starting point.
Source file(s): `arxiv.org/abs/1905.03696` (HAWQ), `arxiv.org/abs/1911.03852` (HAWQ-V2)
Recommendation: Use this to (a) fix CITE-1's uncited claim with a defensible, specific version
("neither paper reports an isolated per-layer validation; both rely on end-to-end accuracy after
allocation" — cite Section III-A/Tables I-IV for HAWQ, Section 2.1/2.3 for HAWQ-V2), and (b)
strengthen the paper's contribution framing in the Introduction.

### CLARIFY-2 — "Methods?"
Source: Prof 2, PDF p3, `/Text` at y≈314, anchored precisely at the start of §4.1 "Setup": *"We
study three architectures, a compact CNN (6 quantizable layers), ResNet-18 (21 layers), and
ResNet-50 (54 layers), on two datasets (CIFAR10, ImageNet100), quantized post-training (PTQ)
with PoT weight encoding, six architecture×dataset combinations in total. All numbers below are
for PTQ."*
Category: **NEEDS-CLARIFICATION**
Finding: Verbatim: *"Methods?"* — single word. Plausibly asking whether this setup description
belongs in §3 Methods rather than opening §4 Results, but per instructions this is not assumed.
Reported verbatim with exact location.
Source file(s): `FSP_B_AXIOM_WORKSHOP_2026_TE.pdf`, page 3.
Recommendation: Author confirms directly against the PDF (page 3, right column, ~top third)
before this is actioned — if the intended meaning is "move to Methods," that's a one-paragraph
relocation, NO-DATA; if it means something else, it needs re-reading in context.

### RANK-2 — "smaller models correlate better?"
Source: Prof 2, PDF p4, `/Text` at Table 2 CNN row
Category: **COMPUTABLE-NOW**
Finding: *"Bei kleineren Modellen scheint es mit wenigen Ausnahmen etwas besser zu
funktionieren, oder?"* Tabulated $S_{raw}$ ρ against $n_{layers}$ across the 6 PTQ combos:

| model | dataset | n_layers | $S_{raw}$ ρ | p |
|---|---|---|---|---|
| cnn | CIFAR10 | 6 | 0.928 | 0.0077 |
| cnn | IMAGENET100 | 6 | 0.600 | 0.208 |
| resnet18 | IMAGENET100 | 21 | 0.121 | 0.600 |
| resnet18 | CIFAR10 | 21 | 0.292 | 0.199 |
| resnet50 | CIFAR10 | 54 | 0.205 | 0.137 |
| resnet50 | IMAGENET100 | 54 | 0.299 | 0.028 |

$\mathrm{spearman}(n_{layers}, \rho) = -0.598$, $p=0.21$ (n=6). **Verdict: directionally
consistent with Prof 2's hunch (negative — smaller n tends toward higher ρ), but not
statistically distinguishable from noise at only 6 data points** — resnet50/IMAGENET100 (n=54)
actually beats resnet18/IMAGENET100 (n=21), breaking strict monotonicity.
Source file(s): both `weight_ablation_canonical_correlation_v2.csv` files, `predictor=raw_trh`
rows, PTQ →`results/review_response/csv/correlation_vs_model_size.csv`
Recommendation: A one-sentence honest answer in the text: directionally yes, but n=6 is too few
architecture×dataset points to claim a trend — don't overclaim this in the resubmission.

### D-1 — Conclusion: wish for a ResNet-50/CIFAR10 finding
Source: Prof 2, PDF p5, `/Text` at the Conclusion, near "...where total damage is small and
diffuse (ResNet-50/CIFAR10), no predictor we test performs well."
Category: **NO-DATA**
Finding: *"Es wäre schon schön, hier noch was rausgefunden zu haben."* (It would be nice to have
found something here.) Not in the original task brief. This reads as an editorial wish, not an
actionable data request — there's no new analysis this maps to; it's asking for a positive
result the null-result data doesn't support.
Recommendation: No new computation. If anything, this strengthens the case for leaning into the
diffuse-damage null result as a genuine finding (as §4.2's text already frames it: "a usable
curvature signal requires damage concentrated enough to separate from noise") rather than trying
to manufacture a positive result under deadline pressure.

---

## Part 3 — Investigation only (no rerun executed)

### 3.1 — Accuracy → loss (Prof 1 mail, primary; Prof 2 PDF B-1/B-2, same ask)

**(a) Is loss already computed anywhere, just discarded?**

**No — genuinely nowhere for the per-layer isolated evaluation.** Traced the exact call path:

- `weight_ablation_canonical.py`'s `_run_isolation_sweep()` (`src/analysis/weight_ablation_canonical.py:249-280`)
  calls `evaluate()` imported from `src/main.py:1246`. That function computes **only** accuracy
  (`correct/total`) — no `criterion`, no loss term anywhere in its body, `torch.no_grad()`
  throughout.
- A **different** function, `_evaluate()` (`src/model_cnn/train.py:235-253`), does compute both
  loss and accuracy (`nn.CrossEntropyLoss`, mean-reduced), and is used for the full-model
  PTQ/QAT evaluation in `src/main.py:529` and `:900` — but this is a *different, non-isolated*
  eval path (whole quantized model, not one-layer-isolated), and even there `ptq_loss` is
  computed and then **discarded**: it is not written to `classification_metrics.csv` (columns
  are `model,dataset,stage,accuracy,precision,recall,f1` only — confirmed by reading the file
  header) or any other CSV.
- Grepped `experiment_log.txt` (2.1M lines) for the canonical CIFAR10 run: `loss` appears 10
  times total in the whole file, none inside a `WeightAblationCanonical`-tagged line. Nothing
  is sitting unused in a log file either.

**Verdict: (ii)** — a genuine rerun, not a text edit, but a cheap one to add once the code is
touched.

**(b) Cost estimate.**

The isolation sweep already re-runs `evaluate()` once per (model, dataset, stage, layer) —
adding a loss accumulation to that same forward pass is essentially free per-call (same pattern
as `_evaluate()` already elsewhere in the codebase). The cost is the **rerun** itself, not the
code change. From the existing canonical run logs (timestamps around the
`[WeightAblationCanonical]`-tagged section):

| dataset | scope | wall time | source |
|---|---|---|---|
| CIFAR10 | PTQ+QAT, 162 isolated evals + 24 gate evals (6 combos) | 00:12:10→00:21:21 = **9m11s** | `results/20260816_230437_38678/logs/experiment_log.txt:8422,8998` |
| IMAGENET100 | PTQ+QAT, 162 isolated evals + 24 gate evals (6 combos) | 09:24:41→11:38:43 = **2h14m2s** | `results/20260816_083054_38677/logs/experiment_log.txt:7281,7856` |

Paper scope is PTQ-only (`neurips_2026.tex:69`, "All numbers below are for PTQ") — restricting
the rerun to PTQ halves the isolated-eval count (81 vs 162 per dataset), so a PTQ-only rerun for
both datasets is roughly **~70–75 minutes of GPU time** (≈4.5min CIFAR10 + ≈67min ImageNet100,
linear-scaling estimate since cost is dominated by one full-val-set forward pass per layer,
repeated). A PTQ+QAT rerun (matching what's already stored for accuracy) is **~2h23min**.

**Recommendation:** This is affordable within a multi-day deadline window if prioritized soon,
but requires: (1) a code change — swap `evaluate()` for a loss-and-accuracy variant inside
`_run_isolation_sweep()`, extend `ABLATION_FIELDNAMES` with an isolated-loss column, extend
`_run_correlations()` to also correlate against a loss-based damage target; (2) then the rerun
itself. Recommend deciding explicitly whether to spend the ~70-min PTQ-only rerun *before*
promising a loss-based damage metric in the resubmission email — do not promise it as a "quick
fix."

### 3.2 — See LIT-1 above (folded in under Prof 2's PDF items since her annotation is the more
specific version of this question).

---

## Part 4 — Needs clarification (verbatim, not interpreted)

See CLARIFY-1 ("Zwischen was?", PDF p1) and CLARIFY-2 ("Methods?", PDF p3) above — both now have
exact page + surrounding-text location (recovered via PDF coordinate mapping), but neither
meaning is assumed. Resolve directly against the PDF or by asking Prof 2 before acting on either.

---

## Files written this pass

```
results/review_response/csv/bootstrap_ci_spearman.csv
results/review_response/csv/normalized_ranks.csv
results/review_response/csv/signed_damage_improving_cases.csv
results/review_response/csv/raw_trace_per_param_comparison.csv
results/review_response/csv/predictor_concordance.csv
results/review_response/csv/correlation_vs_model_size.csv
results/review_response/csv/reconciliation_ledger_resnet50_conv1.csv
results/review_response/csv/intro_citation_audit.csv
```

No file under an existing `results/<RUN_ID>/` directory was modified. `paper/neurips_2026.tex`
was not edited.
