"""
Self-Rating Circularity Experiment — Statistical Analysis (Phase 5)

Implements 5 pre-registered confirmatory tests + 4 exploratory analyses.
Uses scipy/statsmodels for proper statistical testing.

Confirmatory tests (Holm-Bonferroni corrected, α = 0.05):
  Test 1: Wilcoxon signed-rank — rating divergence LLM vs SSR (N=345 pairs)
  Test 2: McNemar — LLM auto-accuracy vs LLM on human text (N=69 pairs)
  Test 3: McNemar — SSR on generated vs human text (N=69 pairs)
  Test 4: Wilcoxon — directional bias differential (N=345 paired errors)
  Test 5: Levene — variance compression of errors (N=345 per group)

Exploratory analyses:
  E1: Kruskal-Wallis on |LLM-SSR| divergence by persona
  E2: Per-domain exact match comparison
  E3: Signed error by target rating level
  E4: SSR confidence on generated vs human text

Usage:
  python research/calibration/analyze-circularity.py [results-file] [baseline-file]

If no files specified, uses the most recent files in research/data/.

Pre-registration date: 2026-02-09
"""

import json
import sys
import os
import glob
import numpy as np
from scipy import stats
from collections import defaultdict

try:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    print("WARNING: statsmodels not installed. McNemar and Holm-Bonferroni unavailable.")
    print("Install with: pip install statsmodels")
    HAS_STATSMODELS = False


def find_latest_file(data_dir: str, prefix: str) -> str | None:
    """Find the most recent file matching a prefix in data dir."""
    pattern = os.path.join(data_dir, f"{prefix}*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    # Filter out partial files
    files = [f for f in files if "partial" not in f]
    return files[0] if files else None


def rank_biserial(w_stat: float, n: int) -> float:
    """Compute rank-biserial correlation from Wilcoxon W statistic.
    r = 1 - (2W) / (n(n+1)/2)
    """
    max_w = n * (n + 1) / 2
    if max_w == 0:
        return 0.0
    return 1 - (2 * w_stat) / max_w


def print_separator(char: str = "=", width: int = 75):
    print(char * width)


def print_test_result(name: str, stat_name: str, stat_val: float,
                      p_raw: float, p_adj: float | None, effect_name: str,
                      effect_val: float, reject: bool | None, extra: str = ""):
    """Format and print a statistical test result."""
    sig = "***" if (p_adj or p_raw) < 0.001 else "**" if (p_adj or p_raw) < 0.01 else "*" if (p_adj or p_raw) < 0.05 else "ns"
    print(f"\n  {name}")
    print(f"    {stat_name} = {stat_val:.4f}")
    print(f"    p (raw)      = {p_raw:.6f}")
    if p_adj is not None:
        print(f"    p (Holm-BF)  = {p_adj:.6f}")
    print(f"    {effect_name} = {effect_val:.4f}")
    if reject is not None:
        print(f"    Reject H0?   = {reject} [{sig}]")
    if extra:
        print(f"    {extra}")


def main():
    # ─── Locate data files ───────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if len(sys.argv) >= 2:
        results_file = sys.argv[1]
    else:
        results_file = find_latest_file(data_dir, "circularity-results-")
        if not results_file:
            print("ERROR: No circularity-results file found in data/")
            print("Run self-rating-experiment.ts first (phases 1-4).")
            sys.exit(1)

    if len(sys.argv) >= 3:
        baseline_file = sys.argv[2]
    else:
        baseline_file = find_latest_file(data_dir, "llm-baseline-as-deployed-")
        if not baseline_file:
            print("WARNING: No LLM baseline file found. Tests 2-3 will be skipped.")

    print_separator()
    print("SELF-RATING CIRCULARITY EXPERIMENT — STATISTICAL ANALYSIS")
    print(f"Pre-registration: 2026-02-09")
    print(f"Results file: {os.path.basename(results_file)}")
    if baseline_file:
        print(f"Baseline file: {os.path.basename(baseline_file)}")
    print_separator()

    # ─── Load data ───────────────────────────────────────────────
    with open(results_file, "r") as f:
        data = json.load(f)

    cases = data["cases"]
    aggregated = data["summary"]["aggregatedByCase"]
    n_cases = len(cases)
    n_agg = len(aggregated)

    print(f"\n  N (individual cases) = {n_cases}")
    print(f"  N (aggregated cases) = {n_agg}")

    # Load baseline for Tests 2-3
    baseline_results = None
    if baseline_file:
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
        baseline_results = {r["label"]: r for r in baseline_data["results"]}
        print(f"  Baseline cases loaded = {len(baseline_results)}")

    # ─── Extract arrays ──────────────────────────────────────────
    llm_ratings = np.array([c["llmRating"] for c in cases])
    ssr_ratings = np.array([c["ssrRating"] for c in cases])
    targets = np.array([c["targetRating"] for c in cases])

    llm_errors = llm_ratings - targets  # signed errors
    ssr_errors = ssr_ratings - targets
    rating_diffs = llm_ratings - ssr_ratings  # LLM - SSR

    print(f"\n  LLM mean rating: {llm_ratings.mean():.3f} (SD={llm_ratings.std():.3f})")
    print(f"  SSR mean rating: {ssr_ratings.mean():.3f} (SD={ssr_ratings.std():.3f})")
    print(f"  Target mean:     {targets.mean():.3f} (SD={targets.std():.3f})")

    # ═══════════════════════════════════════════════════════════════
    # CONFIRMATORY TESTS
    # ═══════════════════════════════════════════════════════════════
    print("\n")
    print_separator()
    print("CONFIRMATORY TESTS (5 pre-registered, Holm-Bonferroni corrected)")
    print_separator()

    raw_pvalues = []
    test_names = []
    test_results_data = []

    # ─── Test 1: Wilcoxon signed-rank (rating divergence) ────────
    # H0: median(LLM_rating - SSR_rating) = 0
    nonzero_diffs = rating_diffs[rating_diffs != 0]
    if len(nonzero_diffs) > 0:
        w1_stat, w1_p = stats.wilcoxon(rating_diffs, alternative="two-sided")
        r1 = rank_biserial(w1_stat, len(nonzero_diffs))
    else:
        w1_stat, w1_p, r1 = 0.0, 1.0, 0.0

    raw_pvalues.append(w1_p)
    test_names.append("Test 1: Rating divergence (Wilcoxon)")
    test_results_data.append({
        "test": "Wilcoxon signed-rank",
        "hypothesis": "H0: median(LLM - SSR) = 0",
        "W": float(w1_stat), "p_raw": float(w1_p),
        "rank_biserial_r": float(r1),
        "n_nonzero": int(len(nonzero_diffs)),
        "mean_diff": float(rating_diffs.mean()),
        "sd_diff": float(rating_diffs.std()),
    })

    # ─── Test 2: McNemar (LLM auto-accuracy vs LLM on human) ────
    if baseline_results and HAS_STATSMODELS:
        # Build 2×2 contingency table
        # Rows: LLM on generated (correct/incorrect)
        # Cols: LLM on human (correct/incorrect)
        a_both_correct = 0  # Both correct
        b_gen_correct_human_wrong = 0  # Generated correct, human wrong
        c_gen_wrong_human_correct = 0  # Generated wrong, human correct
        d_both_wrong = 0  # Both wrong

        for agg in aggregated:
            label = agg["testCaseLabel"]
            llm_auto_correct = agg["llmExactMatch"]
            if label in baseline_results:
                llm_human_correct = baseline_results[label]["exact"]
                if llm_auto_correct and llm_human_correct:
                    a_both_correct += 1
                elif llm_auto_correct and not llm_human_correct:
                    b_gen_correct_human_wrong += 1
                elif not llm_auto_correct and llm_human_correct:
                    c_gen_wrong_human_correct += 1
                else:
                    d_both_wrong += 1

        contingency_2 = np.array([[a_both_correct, b_gen_correct_human_wrong],
                                   [c_gen_wrong_human_correct, d_both_wrong]])

        # McNemar uses the off-diagonal cells
        b = b_gen_correct_human_wrong
        c = c_gen_wrong_human_correct

        if b + c > 0:
            from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
            result_mc2 = mcnemar_test(contingency_2, exact=(b + c < 25))
            mcnemar2_p = result_mc2.pvalue
            mcnemar2_stat = result_mc2.statistic
        else:
            mcnemar2_p = 1.0
            mcnemar2_stat = 0.0

        odds_ratio_2 = b / c if c > 0 else float('inf') if b > 0 else 1.0

        raw_pvalues.append(mcnemar2_p)
        test_names.append("Test 2: LLM auto-accuracy vs human (McNemar)")
        test_results_data.append({
            "test": "McNemar",
            "hypothesis": "H0: LLM accuracy on own text = LLM accuracy on human text",
            "contingency": contingency_2.tolist(),
            "b_discordant": int(b), "c_discordant": int(c),
            "statistic": float(mcnemar2_stat), "p_raw": float(mcnemar2_p),
            "odds_ratio": float(odds_ratio_2),
            "llm_auto_exact": int(sum(1 for a in aggregated if a["llmExactMatch"])),
            "llm_human_exact": int(sum(1 for r in baseline_results.values() if r["exact"])),
        })
    else:
        raw_pvalues.append(1.0)
        test_names.append("Test 2: LLM auto-accuracy (SKIPPED - no baseline)")
        test_results_data.append({"test": "McNemar", "skipped": True})

    # ─── Test 3: McNemar (SSR on generated vs human) ─────────────
    # Need SSR results on human text — use cross-validation data
    cv_file = find_latest_file(data_dir, "cross-validation-")
    if cv_file and HAS_STATSMODELS:
        # The cross-validation file has LODO results but not per-case predictions.
        # We need to reconstruct per-case SSR predictions from the raw data.
        # Since we don't have per-case SSR on human text in the CV file,
        # we'll use a simpler approach: just compare aggregate exact match rates.
        # NOTE: This is a limitation — without per-case SSR predictions on human text,
        # we can't do a proper paired McNemar test.
        # For now, we'll mark this test as requiring additional data.

        # Actually, let's try to find ablation data which has per-case SSR predictions
        ablation_file = find_latest_file(data_dir, "ablation-v2-")
        if ablation_file:
            with open(ablation_file, "r") as f:
                ablation_data = json.load(f)

            # Find H3 (asymmetric) results which match our SSR config
            h3_results = None
            # Ablation v2 uses "conditions" with "details" per case
            for cond in ablation_data.get("conditions", ablation_data.get("variants", [])):
                name = cond.get("name", "")
                if "H3" in name or "asymm" in name.lower() or "asymmetric" in name.lower():
                    h3_results = cond.get("details", cond.get("perCase", cond.get("results", [])))
                    break

            if h3_results and isinstance(h3_results, list):
                ssr_human_by_label = {}
                for r in h3_results:
                    label = r.get("label", r.get("testCaseLabel", ""))
                    expected = r.get("expected", r.get("targetRating", 0))
                    predicted = r.get("predicted", r.get("ssrRating", 0))
                    ssr_human_by_label[label] = (predicted == expected)

                # Build contingency table
                a3 = b3 = c3 = d3 = 0
                for agg in aggregated:
                    label = agg["testCaseLabel"]
                    ssr_gen_correct = agg["ssrExactMatch"]
                    if label in ssr_human_by_label:
                        ssr_human_correct = ssr_human_by_label[label]
                        if ssr_gen_correct and ssr_human_correct:
                            a3 += 1
                        elif ssr_gen_correct and not ssr_human_correct:
                            b3 += 1
                        elif not ssr_gen_correct and ssr_human_correct:
                            c3 += 1
                        else:
                            d3 += 1

                contingency_3 = np.array([[a3, b3], [c3, d3]])
                if b3 + c3 > 0:
                    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
                    result_mc3 = mcnemar_test(contingency_3, exact=(b3 + c3 < 25))
                    mcnemar3_p = result_mc3.pvalue
                    mcnemar3_stat = result_mc3.statistic
                else:
                    mcnemar3_p = 1.0
                    mcnemar3_stat = 0.0

                odds_ratio_3 = b3 / c3 if c3 > 0 else float('inf') if b3 > 0 else 1.0

                raw_pvalues.append(mcnemar3_p)
                test_names.append("Test 3: SSR generated vs human (McNemar)")
                test_results_data.append({
                    "test": "McNemar",
                    "hypothesis": "H0: SSR accuracy on generated text = SSR accuracy on human text",
                    "contingency": contingency_3.tolist(),
                    "b_discordant": int(b3), "c_discordant": int(c3),
                    "statistic": float(mcnemar3_stat), "p_raw": float(mcnemar3_p),
                    "odds_ratio": float(odds_ratio_3),
                    "ssr_gen_exact": int(sum(1 for a in aggregated if a["ssrExactMatch"])),
                })
            else:
                raw_pvalues.append(1.0)
                test_names.append("Test 3: SSR generated vs human (SKIPPED - no per-case data)")
                test_results_data.append({"test": "McNemar", "skipped": True,
                                          "reason": "No per-case SSR predictions found in ablation data"})
        else:
            raw_pvalues.append(1.0)
            test_names.append("Test 3: SSR generated vs human (SKIPPED - no ablation data)")
            test_results_data.append({"test": "McNemar", "skipped": True})
    else:
        raw_pvalues.append(1.0)
        test_names.append("Test 3: SSR generated vs human (SKIPPED)")
        test_results_data.append({"test": "McNemar", "skipped": True})

    # ─── Test 4: Wilcoxon on signed errors ───────────────────────
    # H0: mean(LLM_error) = mean(SSR_error), where error = rating - target
    paired_error_diffs = llm_errors - ssr_errors
    nonzero_err_diffs = paired_error_diffs[paired_error_diffs != 0]
    if len(nonzero_err_diffs) > 0:
        w4_stat, w4_p = stats.wilcoxon(paired_error_diffs, alternative="two-sided")
        r4 = rank_biserial(w4_stat, len(nonzero_err_diffs))
    else:
        w4_stat, w4_p, r4 = 0.0, 1.0, 0.0

    raw_pvalues.append(w4_p)
    test_names.append("Test 4: Directional bias differential (Wilcoxon)")
    test_results_data.append({
        "test": "Wilcoxon signed-rank",
        "hypothesis": "H0: mean(LLM_error) = mean(SSR_error)",
        "W": float(w4_stat), "p_raw": float(w4_p),
        "rank_biserial_r": float(r4),
        "n_nonzero": int(len(nonzero_err_diffs)),
        "llm_mean_error": float(llm_errors.mean()),
        "ssr_mean_error": float(ssr_errors.mean()),
        "llm_sd_error": float(llm_errors.std()),
        "ssr_sd_error": float(ssr_errors.std()),
    })

    # ─── Test 5: Levene (variance compression) ───────────────────
    # H0: var(LLM_errors) = var(SSR_errors)
    f5_stat, f5_p = stats.levene(llm_errors, ssr_errors, center="median")
    var_ratio = float(np.var(llm_errors)) / float(np.var(ssr_errors)) if np.var(ssr_errors) > 0 else float('inf')

    raw_pvalues.append(f5_p)
    test_names.append("Test 5: Variance compression (Levene)")
    test_results_data.append({
        "test": "Levene",
        "hypothesis": "H0: var(LLM_errors) = var(SSR_errors)",
        "F": float(f5_stat), "p_raw": float(f5_p),
        "variance_ratio": float(var_ratio),
        "llm_var": float(np.var(llm_errors)),
        "ssr_var": float(np.var(ssr_errors)),
        "llm_sd": float(np.std(llm_errors)),
        "ssr_sd": float(np.std(ssr_errors)),
    })

    # ─── Holm-Bonferroni correction ──────────────────────────────
    if HAS_STATSMODELS:
        reject_arr, corrected_p, _, _ = multipletests(raw_pvalues, method="holm")
        for i, td in enumerate(test_results_data):
            td["p_adjusted"] = float(corrected_p[i])
            td["reject_h0"] = bool(reject_arr[i])
    else:
        corrected_p = raw_pvalues
        reject_arr = [p < 0.05 for p in raw_pvalues]

    # ─── Print confirmatory results ──────────────────────────────
    for i, name in enumerate(test_names):
        td = test_results_data[i]
        if td.get("skipped"):
            print(f"\n  {name}")
            print(f"    SKIPPED: {td.get('reason', 'Missing data')}")
            continue

        if "W" in td:
            print_test_result(
                name, "W", td["W"], td["p_raw"],
                td.get("p_adjusted"), "rank-biserial r", td.get("rank_biserial_r", 0),
                td.get("reject_h0"),
                f"N_nonzero={td.get('n_nonzero', 'N/A')}"
            )
        elif "F" in td:
            print_test_result(
                name, "F", td["F"], td["p_raw"],
                td.get("p_adjusted"), "variance ratio", td.get("variance_ratio", 0),
                td.get("reject_h0"),
                f"LLM_var={td['llm_var']:.4f}, SSR_var={td['ssr_var']:.4f}"
            )
        elif "statistic" in td:
            print_test_result(
                name, "chi2/exact", td["statistic"], td["p_raw"],
                td.get("p_adjusted"), "odds ratio", td.get("odds_ratio", 0),
                td.get("reject_h0"),
                f"Discordant: b={td.get('b_discordant')}, c={td.get('c_discordant')}"
            )

    # Summary
    n_significant = sum(1 for td in test_results_data if td.get("reject_h0", False))
    n_tested = sum(1 for td in test_results_data if not td.get("skipped", False))
    print(f"\n  Summary: {n_significant}/{n_tested} tests significant after Holm-Bonferroni (alpha=0.05)")

    # ═══════════════════════════════════════════════════════════════
    # EXPLORATORY ANALYSES
    # ═══════════════════════════════════════════════════════════════
    print("\n")
    print_separator()
    print("EXPLORATORY ANALYSES (4 post-hoc, labeled as exploratory)")
    print_separator()

    # ─── E1: Kruskal-Wallis by persona ───────────────────────────
    print("\n  E1: Persona × Method Interaction")
    print("  " + "-" * 55)

    persona_groups = defaultdict(list)
    for c in cases:
        persona_groups[c["personaId"]].append(abs(c["llmRating"] - c["ssrRating"]))

    groups = [np.array(persona_groups[pid]) for pid in sorted(persona_groups.keys())]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis H = {kw_stat:.4f}, p = {kw_p:.6f}")
    else:
        print("  Not enough groups for Kruskal-Wallis")

    print(f"\n  {'Persona':<25} {'Mean |div|':<12} {'Median |div|':<12} {'N':<5}")
    print("  " + "-" * 55)
    for pid in sorted(persona_groups.keys()):
        arr = np.array(persona_groups[pid])
        print(f"  {pid:<25} {arr.mean():<12.3f} {np.median(arr):<12.1f} {len(arr):<5}")

    # ─── E2: Per-domain divergence table ─────────────────────────
    print("\n  E2: Per-Domain Divergence")
    print("  " + "-" * 65)
    print(f"  {'Domain':<18} {'LLM exact%':<12} {'SSR exact%':<12} {'Mean div':<10} {'N':<5}")
    print("  " + "-" * 65)

    per_domain = data["summary"]["perDomain"]
    for domain in sorted(per_domain.keys()):
        dm = per_domain[domain]
        domain_cases = [c for c in cases if c["domain"] == domain]
        n_domain = len(domain_cases)
        print(f"  {domain:<18} {dm['llmExact']:<12} {dm['ssrExact']:<12} {dm['meanDivergence']:<10.3f} {n_domain:<5}")

    # ─── E3: Signed error by target rating level ─────────────────
    print("\n  E3: Error Distribution by Target Rating Level")
    print("  " + "-" * 65)
    print(f"  {'Rating':<8} {'LLM mean err':<14} {'SSR mean err':<14} {'LLM SD':<10} {'SSR SD':<10} {'N':<5}")
    print("  " + "-" * 65)

    for target in range(1, 6):
        mask = targets == target
        if mask.sum() == 0:
            continue
        llm_e = llm_errors[mask]
        ssr_e = ssr_errors[mask]
        print(f"  {target:<8} {llm_e.mean():<14.3f} {ssr_e.mean():<14.3f} {llm_e.std():<10.3f} {ssr_e.std():<10.3f} {mask.sum():<5}")

    # ─── E4: SSR confidence comparison ───────────────────────────
    print("\n  E4: SSR Confidence on Generated vs Human Text")
    print("  " + "-" * 55)

    ssr_gen_conf = np.array([c["ssrConfidence"] for c in cases])
    print(f"  Generated text: mean={ssr_gen_conf.mean():.3f}, SD={ssr_gen_conf.std():.3f}, median={np.median(ssr_gen_conf):.3f}")

    # If we have human SSR confidence data from ablation
    if ablation_file:
        try:
            with open(ablation_file, "r") as f:
                ablation_data = json.load(f)
            # Try to extract confidence from H3 variant
            for cond in ablation_data.get("conditions", ablation_data.get("variants", [])):
                name = cond.get("name", "")
                if "H3" in name or "asymm" in name.lower() or "asymmetric" in name.lower():
                    h3_cases = cond.get("details", cond.get("perCase", []))
                    if h3_cases:
                        human_confs = [r.get("confidence", 0) for r in h3_cases if "confidence" in r]
                        if human_confs:
                            human_conf_arr = np.array(human_confs)
                            print(f"  Human text:     mean={human_conf_arr.mean():.3f}, SD={human_conf_arr.std():.3f}, median={np.median(human_conf_arr):.3f}")
                            # Mann-Whitney U test
                            u_stat, u_p = stats.mannwhitneyu(ssr_gen_conf, human_conf_arr, alternative="two-sided")
                            print(f"  Mann-Whitney U = {u_stat:.1f}, p = {u_p:.6f}")
                    break
        except Exception as e:
            print(f"  Could not load human SSR confidence: {e}")

    # ═══════════════════════════════════════════════════════════════
    # SAVE COMPLETE STATISTICAL RESULTS
    # ═══════════════════════════════════════════════════════════════
    stats_output = {
        "preRegistrationDate": "2026-02-09",
        "resultsFile": os.path.basename(results_file),
        "baselineFile": os.path.basename(baseline_file) if baseline_file else None,
        "nCases": n_cases,
        "nAggregated": n_agg,
        "confirmatoryTests": test_results_data,
        "nSignificant": n_significant,
        "nTested": n_tested,
        "alphaLevel": 0.05,
        "correction": "Holm-Bonferroni",
        "descriptive": {
            "llm_mean_rating": float(llm_ratings.mean()),
            "ssr_mean_rating": float(ssr_ratings.mean()),
            "target_mean": float(targets.mean()),
            "rating_diff_mean": float(rating_diffs.mean()),
            "rating_diff_sd": float(rating_diffs.std()),
            "rating_diff_median": float(np.median(rating_diffs)),
            "llm_exact_pct": float((np.sum(llm_errors == 0) / n_cases) * 100),
            "ssr_exact_pct": float((np.sum(ssr_errors == 0) / n_cases) * 100),
            "llm_within1_pct": float((np.sum(np.abs(llm_errors) <= 1) / n_cases) * 100),
            "ssr_within1_pct": float((np.sum(np.abs(ssr_errors) <= 1) / n_cases) * 100),
            "llm_mae": float(np.abs(llm_errors).mean()),
            "ssr_mae": float(np.abs(ssr_errors).mean()),
        },
        "exploratory": {
            "kruskal_wallis_persona": {
                "H": float(kw_stat) if 'kw_stat' in dir() else None,
                "p": float(kw_p) if 'kw_p' in dir() else None,
            },
            "per_target_rating": {
                str(t): {
                    "llm_mean_error": float(llm_errors[targets == t].mean()),
                    "ssr_mean_error": float(ssr_errors[targets == t].mean()),
                    "n": int((targets == t).sum()),
                }
                for t in range(1, 6) if (targets == t).sum() > 0
            },
        },
    }

    stats_path = os.path.join(data_dir, f"circularity-stats-{os.path.basename(results_file).replace('circularity-results-', '').replace('.json', '')}.json")
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2)

    print(f"\n  Statistical results saved to: {os.path.basename(stats_path)}")

    # ═══════════════════════════════════════════════════════════════
    # SCENARIO DETERMINATION
    # ═══════════════════════════════════════════════════════════════
    print("\n")
    print_separator()
    print("SCENARIO DETERMINATION")
    print_separator()

    if n_significant == 0:
        print("\n  --> SCENARIO 2: No detectable circularity bias")
        print("      All confirmatory tests non-significant after correction.")
        print("      Narrative: Null result — circularity does not produce detectable")
        print("      bias in these domains with this model.")
    elif n_significant >= 3:
        print("\n  --> SCENARIO 1: Circularity bias detected")
        print(f"      {n_significant}/{n_tested} tests significant.")
        print("      Narrative: Self-rating produces measurable bias. SSR provides")
        print("      necessary independence guarantee.")
    else:
        print("\n  --> SCENARIO 3: Mixed / domain-specific results")
        print(f"      {n_significant}/{n_tested} tests significant.")
        print("      Narrative: Circularity effects are real but moderate/context-dependent.")
        print("      Recommendation: Use both methods, flag disagreements.")

    print_separator()
    print("Analysis complete.")
    print_separator()


if __name__ == "__main__":
    main()
