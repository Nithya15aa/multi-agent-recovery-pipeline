"""
Optimized Evaluation Harness v2
===============================
Reduces API calls by ~40-60% vs original through:
  1. Parallel baseline execution (A, B, majority run concurrently)
  2. Resume from checkpoint (skip already-evaluated samples)
  3. Merged verify+classify into single LLM call (saves 1 call per failure)
  4. Shared first-pass output (baseline A reuses pipeline's first attempt)
  5. Configurable: skip any baseline you already have data for

API calls per sample comparison:
  Original: ~12 calls (all baselines + pipeline + step attribution)
  Optimized: ~5-7 calls (shared execution, merged prompts, parallel baselines)
"""

import csv
import os
import json
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .agents import (
    PipelineOrchestrator, LLMBackend,
    normalize_numeric, normalize_letter, normalize_yesno,
)



def compute_metrics(results, n):
    ba = sum(1 for r in results if r["baseline_a_correct"])
    bb = sum(1 for r in results if r.get("baseline_b_correct", False))
    mv = sum(1 for r in results if r.get("majority_correct", False))
    pp = sum(1 for r in results if r["pipeline_correct"])

    tp = fp = tn = fn = 0
    for r in results:
        fa_ok = r["first_attempt_correct"]
        fl    = r["verif_flagged_invalid"]
        if fl and not fa_ok:     tp += 1
        elif fl and fa_ok:       fp += 1
        elif not fl and fa_ok:   tn += 1
        else:                    fn += 1

    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    fdr  = rec
    fpr  = fp/(fp+tn) if (fp+tn) else 0.0

    rsr_n = sum(1 for r in results if r["recovery_triggered"] and r["pipeline_correct"])
    rsr_d = sum(1 for r in results if r["recovery_triggered"])
    rsr   = rsr_n/rsr_d if rsr_d else 0.0

    tot_att = sum(r["pipeline_attempts"] for r in results)

    ft_counts = defaultdict(int)
    ft_recov  = defaultdict(int)
    for r in results:
        for ft in r.get("failure_types", []):
            ft_counts[ft] += 1
            if r["pipeline_correct"]:
                ft_recov[ft] += 1

    surg_ct = sum(1 for r in results if r.get("surgical_used", False))
    surg_ok = sum(1 for r in results if r.get("surgical_used", False) and r["pipeline_correct"])

    return dict(
        n=n, baseline_a_tsr=ba/n, baseline_b_tsr=bb/n, majority_tsr=mv/n,
        pipeline_tsr=pp/n, tsr_improvement=(pp-ba)/n,
        fdr=fdr, fpr=fpr, precision=prec, recall=rec, f1=f1,
        verif_tp=tp, verif_fp=fp, verif_tn=tn, verif_fn=fn,
        rsr=rsr, rsr_num=rsr_n, rsr_den=rsr_d, aat=tot_att/n if n else 0,
        failure_type_counts=dict(ft_counts),
        failure_type_recovered=dict(ft_recov),
        surgical_attempts=surg_ct, surgical_recovered=surg_ok,
    )


def bootstrap_ci(values, stat_fn=np.mean, n_boot=10000, ci=0.95):
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.array(values)
    bs = sorted(stat_fn(np.random.choice(arr, len(arr), True)) for _ in range(n_boot))
    a = (1-ci)/2
    return (stat_fn(arr), bs[int(a*n_boot)], bs[int((1-a)*n_boot)])


def format_gsm8k(sample):
    gt = sample["answer"].split("####")[-1].strip().replace(",","").replace("$","").replace("%","").strip()
    return sample["question"], gt

def format_arc(sample):
    choices = sample["choices"]["text"]
    labels  = sample["choices"]["label"]
    cstr = "\n".join(f"{l}. {t}" for l, t in zip(labels, choices))
    q = f"{sample['question']}\n\nChoices:\n{cstr}\n\nAnswer with ONLY the letter (A, B, C, or D)."
    km = {"1":"A","2":"B","3":"C","4":"D"}
    ak = km.get(sample["answerKey"].strip(), sample["answerKey"].strip()).upper()
    return q, ak

def format_boolq(sample):
    q = (f"Passage: {sample['passage']}\n\nQuestion: {sample['question']}\n\n"
         f"Answer with ONLY 'yes' or 'no'.")
    return q, "yes" if sample["answer"] else "no"

FORMATTERS  = {"gsm8k": format_gsm8k, "arc": format_arc, "boolq": format_boolq}
NORMALIZERS = {"gsm8k": normalize_numeric, "arc": normalize_letter, "boolq": normalize_yesno}


def load_checkpoint(csv_path):
    """Load already-evaluated samples from CSV to enable resume."""
    if not csv_path or not os.path.exists(csv_path):
        return [], 0

    results = []
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string booleans back
                for key in ["baseline_a_correct", "baseline_b_correct", "majority_correct",
                            "first_attempt_correct", "verif_flagged_invalid",
                            "pipeline_correct", "recovery_triggered", "recovery_succeeded",
                            "surgical_used"]:
                    if key in row:
                        row[key] = row[key].strip().lower() in ("true", "1", "yes")

                for key in ["pipeline_attempts", "index", "failure_memory_size"]:
                    if key in row:
                        try:
                            row[key] = int(row[key])
                        except (ValueError, TypeError):
                            row[key] = 0

                # Parse failure types
                ft_raw = row.get("failure_types", "NONE")
                row["failure_types"] = [ft for ft in ft_raw.split("|") if ft and ft != "NONE"]

                results.append(row)
    except Exception as e:
        print(f"  Warning: Could not load checkpoint: {e}")
        return [], 0

    n_done = len(results)
    print(f"  Checkpoint loaded: {n_done} samples already evaluated")
    return results, n_done


class EvaluationRunner:
    """
    Optimized runner with:
    - Parallel baseline execution
    - Checkpoint resume
    - Shared first-pass output between baseline A and pipeline
    - Configurable baseline skipping
    """

    def __init__(self, llm, domain, vectorizer, rag_fn=None, config_name="full_pipeline"):
        self.llm = llm
        self.domain = domain
        self.vectorizer = vectorizer
        self.rag_fn = rag_fn
        self.config_name = config_name
        self.fmt  = FORMATTERS[domain]
        self.norm = NORMALIZERS[domain]

    def run(self, dataset, n_samples, csv_path=None, rsr_csv_path=None,
            run_baselines=True, run_majority=False, pipeline_kwargs=None,
            resume=True, parallel_baselines=True, majority_k=3):
        """
        Args:
            dataset:             HF dataset or list of samples
            n_samples:           Number of samples to evaluate
            csv_path:            Path to save per-sample CSV results
            rsr_csv_path:        Path to save RSR timeline CSV
            run_baselines:       Whether to run baseline A and B
            run_majority:        Whether to run majority vote baseline
            pipeline_kwargs:     Override pipeline config (for ablations)
            resume:              If True, skip already-evaluated samples
            parallel_baselines:  If True, run baselines in parallel threads
            majority_k:          Number of samples for majority vote (default 3)
        """
        if pipeline_kwargs is None:
            pipeline_kwargs = {}

        pipe = PipelineOrchestrator(
            llm=self.llm, domain=self.domain,
            vectorizer=self.vectorizer, rag_fn=self.rag_fn,
            **pipeline_kwargs,
        )

        # ── Resume from checkpoint ──
        existing_results = []
        start_idx = 0
        if resume and csv_path:
            existing_results, start_idx = load_checkpoint(csv_path)

        results = list(existing_results)
        rsr_tl = []
        rsr_n = sum(1 for r in existing_results if r.get("recovery_triggered") and r.get("pipeline_correct"))
        rsr_d = sum(1 for r in existing_results if r.get("recovery_triggered"))
        n = min(n_samples, len(dataset))

        # Rebuild RSR timeline from existing results
        for i, r in enumerate(existing_results):
            rsr_tl.append({
                "sample": i + 1,
                "rsr": rsr_n / rsr_d if rsr_d else 0.0,
                "memory_size": r.get("failure_memory_size", 0),
            })

        if start_idx >= n:
            print(f"  All {n} samples already evaluated. Skipping to metrics.")
            metrics = compute_metrics(results, n)
            metrics["pipeline_state"] = pipe.get_state()
            metrics["config_name"] = self.config_name
            metrics["domain"] = self.domain
            return metrics, results, rsr_tl

        print(f"\n  Evaluating samples {start_idx + 1} to {n} "
              f"({n - start_idx} remaining, {start_idx} cached)")

        # ── Estimate API calls ──
        calls_per_sample = 1  # pipeline execution (minimum)
        if run_baselines: calls_per_sample += 3  # baseline A (1) + baseline B (2)
        if run_majority: calls_per_sample += majority_k
        calls_per_sample += 2  # pipeline verify + potential recovery
        remaining = n - start_idx
        print(f"  Estimated API calls: ~{remaining * calls_per_sample} "
              f"({calls_per_sample}/sample × {remaining} samples)")

        fields = [
            "index", "question_preview", "ground_truth",
            "baseline_a_output", "baseline_a_correct",
            "baseline_b_output", "baseline_b_correct",
            "majority_output", "majority_correct",
            "first_attempt_output", "first_attempt_correct",
            "verif_flagged_invalid", "failure_types",
            "pipeline_output", "pipeline_attempts", "pipeline_correct",
            "recovery_triggered", "recovery_succeeded",
            "surgical_used", "recoverability_scores", "failure_memory_size",
        ]

        # Open CSV in append mode if resuming, write mode if fresh
        mode = "a" if (resume and start_idx > 0) else "w"
        fh = open(csv_path, mode, newline="", encoding="utf-8") if csv_path else None
        wr = csv.DictWriter(fh, fieldnames=fields) if fh else None
        if wr and mode == "w":
            wr.writeheader()

        try:
            for i in range(start_idx, n):
                q, gt = self.fmt(dataset[i])
                print(f"\n[{i + 1}/{n}] {q[:80]}...")
                print(f"  GT: {gt} | Memory: {pipe.memory.size}")

                # ── Run baselines (optionally in parallel) ──
                ba_out = ""; ba_ok = False
                bb_out = ""; bb_ok = False
                mv_out = ""; mv_ok = False

                if parallel_baselines and (run_baselines or run_majority):
                    ba_out, ba_ok, bb_out, bb_ok, mv_out, mv_ok = \
                        self._run_baselines_parallel(
                            pipe, q, gt, run_baselines, run_majority, majority_k)
                else:
                    if run_baselines:
                        ba_out = pipe.run_baseline_a(q)
                        ba_ok = (ba_out == self.norm(gt))
                        print(f"  Baseline A: {ba_out} → {'✓' if ba_ok else '✗'}")
                        self.llm.delay()

                        bb_out, _ = pipe.run_baseline_b(q, gt)
                        bb_ok = (bb_out == self.norm(gt))
                        print(f"  Baseline B: {bb_out} → {'✓' if bb_ok else '✗'}")
                        self.llm.delay()

                    if run_majority:
                        mv_out = pipe.run_majority_vote(q, k=majority_k)
                        mv_ok = (mv_out == self.norm(gt))
                        print(f"  Majority(k={majority_k}): {mv_out} → {'✓' if mv_ok else '✗'}")
                        self.llm.delay()

                # ── Run pipeline ──
                res = pipe.run_pipeline(q, ground_truth=gt)
                p_out = res["output"]
                p_ok = (p_out == self.norm(gt))
                fa_ok = (res["first_attempt_ans"] == self.norm(gt))
                fl = res["verdicts_log"][0]["flagged_invalid"] if res["verdicts_log"] else False
                rec_trig = not fa_ok and fl
                rec_ok = rec_trig and p_ok
                print(f"  Pipeline: {p_out} (att={res['attempts']}) → {'✓' if p_ok else '✗'}")

                # ── Optimization: reuse pipeline first attempt as baseline A ──
                # If we didn't run baselines separately, use pipeline's first pass
                if not run_baselines:
                    ba_out = res["first_attempt_ans"]
                    ba_ok = fa_ok

                for ft in res["failure_types"]:
                    pipe.update_estimator(ft, p_ok)

                if not fa_ok:
                    rsr_d += 1
                    if p_ok:
                        rsr_n += 1
                rsr_tl.append({
                    "sample": i + 1,
                    "rsr": rsr_n / rsr_d if rsr_d else 0.0,
                    "memory_size": pipe.memory.size,
                })

                rec = dict(
                    index=i, question=q, ground_truth=gt,
                    baseline_a_output=ba_out, baseline_a_correct=ba_ok,
                    baseline_b_output=bb_out, baseline_b_correct=bb_ok,
                    majority_output=mv_out, majority_correct=mv_ok,
                    first_attempt_ans=res["first_attempt_ans"],
                    first_attempt_correct=fa_ok, verif_flagged_invalid=fl,
                    failure_types=res["failure_types"],
                    pipeline_output=p_out, pipeline_attempts=res["attempts"],
                    pipeline_correct=p_ok, recovery_triggered=rec_trig,
                    recovery_succeeded=rec_ok, surgical_used=res["surgical_used"],
                    recoverability_scores=res["recoverability_scores"],
                    memory_size=res["memory_size"],
                )
                results.append(rec)

                if wr:
                    wr.writerow({
                        "index": i,
                        "question_preview": q[:100],
                        "ground_truth": gt,
                        "baseline_a_output": ba_out,
                        "baseline_a_correct": ba_ok,
                        "baseline_b_output": bb_out,
                        "baseline_b_correct": bb_ok,
                        "majority_output": mv_out,
                        "majority_correct": mv_ok,
                        "first_attempt_output": res["first_attempt_ans"],
                        "first_attempt_correct": fa_ok,
                        "verif_flagged_invalid": fl,
                        "failure_types": "|".join(res["failure_types"]) or "NONE",
                        "pipeline_output": p_out,
                        "pipeline_attempts": res["attempts"],
                        "pipeline_correct": p_ok,
                        "recovery_triggered": rec_trig,
                        "recovery_succeeded": rec_ok,
                        "surgical_used": res["surgical_used"],
                        "recoverability_scores": str(res["recoverability_scores"]),
                        "failure_memory_size": res["memory_size"],
                    })
                    fh.flush()
                self.llm.delay()
        finally:
            if fh:
                fh.close()

        if rsr_csv_path:
            with open(rsr_csv_path, "w", newline="", encoding="utf-8") as rf:
                w2 = csv.DictWriter(rf, fieldnames=["sample", "rsr", "memory_size"])
                w2.writeheader()
                w2.writerows(rsr_tl)

        metrics = compute_metrics(results, n)
        metrics["pipeline_state"] = pipe.get_state()
        metrics["config_name"] = self.config_name
        metrics["domain"] = self.domain
        return metrics, results, rsr_tl

    def _run_baselines_parallel(self, pipe, q, gt, run_baselines, run_majority, k=3):
        """Run baselines concurrently using threads. Saves wall-clock time."""
        ba_out = ""; ba_ok = False
        bb_out = ""; bb_ok = False
        mv_out = ""; mv_ok = False

        futures = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            if run_baselines:
                futures["ba"] = executor.submit(pipe.run_baseline_a, q)
                futures["bb"] = executor.submit(pipe.run_baseline_b, q, gt)
            if run_majority:
                futures["mv"] = executor.submit(pipe.run_majority_vote, q, k)

            for key, future in futures.items():
                try:
                    result = future.result(timeout=120)
                    if key == "ba":
                        ba_out = result
                        ba_ok = (ba_out == self.norm(gt))
                        print(f"  Baseline A: {ba_out} → {'✓' if ba_ok else '✗'}")
                    elif key == "bb":
                        bb_out, _ = result
                        bb_ok = (bb_out == self.norm(gt))
                        print(f"  Baseline B: {bb_out} → {'✓' if bb_ok else '✗'}")
                    elif key == "mv":
                        mv_out = result
                        mv_ok = (mv_out == self.norm(gt))
                        print(f"  Majority(k={k}): {mv_out} → {'✓' if mv_ok else '✗'}")
                except Exception as e:
                    print(f"  Warning: {key} baseline failed: {e}")

        return ba_out, ba_ok, bb_out, bb_ok, mv_out, mv_ok


class MajorityOnlyRunner:
    """
    Ultra-lightweight runner that ONLY computes majority vote.
    Use when you already have pipeline + baseline results and just
    need to add the majority vote column.

    Reads existing CSV, adds majority_output and majority_correct,
    writes updated CSV. Costs exactly 3 API calls per sample.
    """

    def __init__(self, llm, domain, vectorizer):
        self.llm = llm
        self.domain = domain
        self.norm = NORMALIZERS[domain]
        self.fmt = FORMATTERS[domain]
        self.vectorizer = vectorizer

    def run(self, dataset, existing_csv_path, output_csv_path=None, n_samples=None, k=3):
        """
        Add majority vote to existing results.
        Reads existing CSV, runs majority vote for each sample,
        updates the majority columns, writes to output_csv_path.
        """
        if output_csv_path is None:
            output_csv_path = existing_csv_path

        # Load existing results
        rows = []
        with open(existing_csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        n = min(n_samples or len(rows), len(rows), len(dataset))
        print(f"\n  Adding majority vote (k={k}) to {n} samples")
        print(f"  Estimated API calls: {n * k}")

        pipe = PipelineOrchestrator(
            llm=self.llm, domain=self.domain,
            vectorizer=self.vectorizer,
        )

        for i in range(n):
            q, gt = self.fmt(dataset[i])

            # Skip if already has majority vote data
            existing_mv = rows[i].get("majority_correct", "").strip().lower()
            if existing_mv in ("true", "false"):
                continue

            mv_out = pipe.run_majority_vote(q, k=k)
            mv_ok = (mv_out == self.norm(gt))
            rows[i]["majority_output"] = mv_out
            rows[i]["majority_correct"] = mv_ok
            print(f"  [{i + 1}/{n}] Majority: {mv_out} → {'✓' if mv_ok else '✗'}")
            self.llm.delay()

        # Write updated CSV
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Recompute metrics
        mv_correct = sum(1 for r in rows[:n]
                         if str(r.get("majority_correct", "")).strip().lower() in ("true", "1"))
        print(f"\n  Majority vote TSR: {mv_correct}/{n} = {mv_correct/n:.1%}")
        return rows


_BASE = dict(
    max_retries=2, enable_step_attribution=True,
    enable_recoverability=True, enable_taxonomy_evolution=False,
    enable_rag_memory=True, random_classification=False,
    use_ground_truth_verification=True,
)

def _cfg(**overrides):
    c = dict(_BASE); c.update(overrides); return c

ABLATION_CONFIGS = {
    "full_pipeline":       _cfg(),
    "original_paper":      _cfg(enable_step_attribution=False, enable_recoverability=False),
    "A1_no_rag":           _cfg(enable_rag_memory=False),
    "A2_random_classify":  _cfg(random_classification=True),
    "A4_retries_1":        _cfg(max_retries=1),
    "A4_retries_3":        _cfg(max_retries=3),
    "A5_no_ground_truth":  _cfg(use_ground_truth_verification=False),
    "A6_smfo":             _cfg(enable_taxonomy_evolution=True),
    "no_step_attribution": _cfg(enable_step_attribution=False),
    "no_recoverability":   _cfg(enable_recoverability=False),
}


def print_metrics_table(metrics):
    W = 64
    d = metrics.get("domain", "?"); c = metrics.get("config_name", "?")
    print(f"\n{'=' * W}\n  Domain: {d} | Config: {c}\n{'=' * W}")
    def row(l, v): print(f"  {l:<36} {v:>20}")
    row("Samples",         str(metrics["n"]))
    row("Baseline A TSR",  f"{metrics['baseline_a_tsr']:.2%}")
    row("Baseline B TSR",  f"{metrics['baseline_b_tsr']:.2%}")
    row("Majority TSR",    f"{metrics['majority_tsr']:.2%}")
    row("Pipeline TSR",    f"{metrics['pipeline_tsr']:.2%}")
    row("TSR Improvement", f"{metrics['tsr_improvement']:+.2%}")
    print(f"  {'-' * 56}")
    row("FDR",  f"{metrics['fdr']:.2%}"); row("FPR", f"{metrics['fpr']:.2%}")
    row("F1",   f"{metrics['f1']:.4f}")
    row("RSR",  f"{metrics['rsr']:.2%}  ({metrics['rsr_num']}/{metrics['rsr_den']})")
    row("AAT",  f"{metrics['aat']:.2f}")
    row("Surgical Attempts", str(metrics["surgical_attempts"]))
    row("Surgical Recovered", str(metrics["surgical_recovered"]))
    print(f"  {'-' * 56}")
    for ft, ct in sorted(metrics["failure_type_counts"].items()):
        rc = metrics["failure_type_recovered"].get(ft, 0)
        print(f"    {ft:<24} {ct:>3} failures | RSR {rc}/{ct}")
    st = metrics.get("pipeline_state", {})
    if st:
        row("Total LLM Calls", str(st.get("total_llm_calls", "?")))
        row("Memory Size",     str(st.get("memory_size", "?")))
    print(f"{'=' * W}\n")