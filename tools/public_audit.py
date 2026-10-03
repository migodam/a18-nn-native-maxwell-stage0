"""Verify the public snapshot and replay saved-label diagnostics, without physics calls.

This adapter is separate from the archived full Stage 0 replay. It cannot
reconstruct labels from missing full-current data or verify unavailable originals.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def jl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_files():
    manifest = json.loads((ROOT / "PUBLIC_MANIFEST.json").read_text())
    found = set()
    for item in manifest["files"]:
        rel = Path(item["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        assert item["path"] not in found
        found.add(item["path"])
        path = ROOT / rel
        assert path.is_file() and not path.is_symlink(), str(rel)
        assert path.stat().st_size == item["bytes"], str(rel)
        assert sha(path) == item["sha256"], str(rel)
        if path.suffix in {".json", ".jsonl", ".csv", ".md", ".log", ".txt", ".tex", ".py", ".svg"}:
            value = path.read_text()
            # A file containing a pattern used by this checker is not a secret.
            patterns = (r"/(?:Volumes|Users)/", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                        r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
                        r"sk-(?:proj-)?[A-Za-z0-9_-]{30,}", r"192\.168\.\d+\.\d+")
            for pattern in patterns:
                assert re.search(pattern, value) is None, (str(rel), "private literal")
    return {"status": "PASS", "manifest_sha256": sha(ROOT / "PUBLIC_MANIFEST.json"), "files_rehashed": len(found)}


def close(actual, archived, context, atol=1e-12):
    assert math.isclose(float(actual), float(archived), rel_tol=2e-8, abs_tol=atol), (context, actual, archived)


def audit_A():
    queries = jl(ROOT / "A18_QUERY_MANIFEST.jsonl")
    table = rows(ROOT / "A18_ANALYTIC_SHORTLIST_BASELINES.csv")
    summaries = rows(ROOT / "results/A18_BASELINE_OBJECT_SUMMARY.csv")
    assert len(queries) == 36 and len({q["query_id"] for q in queries}) == 36
    incoming = {}; total = feasible = labels = 0
    for q in queries:
        items = jl(ROOT / "results" / f"incoming_{q['query_id'][:16]}.jsonl")
        ids = [x["incoming_id"] for x in items]
        assert len(ids) == len(set(ids))
        assert set(ids) == set(range(q["dictionary_atoms"])) - set(q["original_atom_ids"])
        assert all(x["total_drop_count"] == len(q["original_atom_ids"]) for x in items)
        assert sum(x["total_drop_count"] for x in items) == q["source_rows"]
        assert sum(x["feasible_drop_count"] for x in items) == q["feasible_moves"]
        total += q["source_rows"]; feasible += q["feasible_moves"]; labels += len(items)
        incoming[q["query_id"]] = items
    assert (total, feasible, labels) == (47712, 43108, 5208)
    qmap = {q["query_id"]: q for q in queries}
    for row in table:
        items = incoming[row["query_id"]]; tau = qmap[row["query_id"]]["tau"]
        pool = json.loads(row["incoming_ids"]); by_id = {r["incoming_id"]: r for r in items}
        assert set(pool) <= set(by_id)
        best = max((r["teacher_utility"] for r in items), default=0.)
        chosen = max((by_id[i]["teacher_utility"] for i in pool), default=0.)
        close(best, row["full_best_positive_gain"], "A full gain", 1e-18)
        close(chosen, row["shortlist_best_positive_gain"], "A shortlist gain", 1e-18)
        close(best-chosen, row["gain_regret"], "A regret", 1e-18)
        positive = [r for r in items if r["teacher_utility"] > tau]
        tied = {r["incoming_id"] for r in positive if best-r["teacher_utility"] <= tau}
        ranked = sorted(positive, key=lambda r: (-r["teacher_utility"], r["incoming_id"]))[:4]
        top4 = {r["incoming_id"] for r in positive if ranked and r["teacher_utility"] >= ranked[-1]["teacher_utility"]-tau}
        close(int(bool(set(pool) & tied)), row["top1_recall"], "A recall1")
        close(len(set(pool) & top4)/len(top4), row["top4_recall"], "A recall4")
        assert row["no_positive_above_floor"] == str(not positive)
    for saved in summaries:
        group = [r for r in table if r["method"] == saved["method"] and r["seed"] == saved["seed"]]
        object_cov = []
        for oid in (2001, 2007, 2012, 2014):
            rr = [r for r in group if r["object_id"] == str(oid)]
            cov = sum(float(r["shortlist_best_positive_gain"]) for r in rr) / sum(float(r["full_best_positive_gain"]) for r in rr)
            object_cov.append(cov)
            if saved["object_id"] == str(oid):
                close(cov, saved["Cov12"], "A object coverage")
        if saved["object_id"] == "EQUAL_OBJECT_MEAN":
            close(sum(object_cov)/4, saved["Cov12"], "A equal-object coverage")
    return {"queries": 36, "complete_incoming_tables": 36, "incoming_labels": labels,
            "total_drop_add_positions": total, "feasible_positions": feasible,
            "baseline_rows_recomputed_from_saved_utilities": len(table), "object_summary_rows": len(summaries),
            "scope": "Saved incoming teachers and recorded shortlists; no rerun of upstream endpoint solves or Qhat ranking."}


def audit_B():
    import numpy as np
    from scipy import linalg as la
    cfg = json.loads((ROOT / "stage0_config.json").read_text())
    splits = json.loads((ROOT / "A18_SPLITS.json").read_text())["folds"]
    directions = jl(ROOT / "results/A18_DIRECTION_MANIFEST.jsonl")
    cohort = [r for r in directions if r["object_id"] in (2001, 2007) and r["exact_u_available"]]
    assert len(directions) == 205 and len(cohort) == 204
    arrays = {}
    for s in directions:
        p = ROOT / s["path"]
        assert sha(p) == s["file_sha256"]
        with np.load(p, allow_pickle=False) as z:
            a = {key: z[key].copy() for key in z.files}
        for base in ("U", "A"):
            assert np.array_equal(a["zF"]-a["z"+base], a["defect"+base])
        q = -a["u"]@a["zF"]-.5*(a["zF"]@a["zF"])+s["material_gain_term"]
        close(q, s["Q_true"], "full directional gain", 1e-13)
        arrays[s["index"]] = a
    fitted = {}; split_map = {r["fold"]: r for r in splits}; fits = 0
    for fold in splits:
        train = [s for s in cohort if s["object_id"] in fold["train_objects"]]
        if not train:
            assert fold["fold"] == 0
            continue
        H = np.column_stack([arrays[s["index"]]["source_sketch"] for s in train])
        rms = np.sqrt(np.mean(H*H, axis=1)); rms = np.maximum(rms, max(float(np.max(rms))*1e-12, 1e-300))
        hn = H/rms[:, None]; alpha = cfg["ridge_relative_alpha"]*float(la.svdvals(hn)[0]**2)
        factor = la.cho_factor(hn.T@hn+alpha*np.eye(len(train)), lower=True)
        for base in ("U", "A"):
            T = np.column_stack([arrays[s["index"]]["defect"+base] for s in train])
            left, _, _ = la.svd(T, full_matrices=False)
            for rank in cfg["pod_ranks"]:
                U = left[:, :rank]; W = (U.T@T)@la.cho_solve(factor, hn.T)
                fitted[fold["fold"], base, rank] = (U, W, rms)
                with np.load(ROOT / "results" / f"fit_fold{fold['fold']}_{base}_rank{rank}.npz") as saved:
                    assert np.array_equal(saved["train_direction_indices"], [s["index"] for s in train])
                    # Singular vectors have a sign/gauge ambiguity. Compare their span.
                    assert np.min(la.svdvals(U.T@saved["basis"])) > 1-1e-10
                    h = arrays[train[0]["index"]]["source_sketch"]
                    assert np.allclose(U@(W@(h/rms)), saved["basis"]@(saved["ridge_weights"]@(h/saved["rms"])), rtol=2e-8, atol=1e-12)
                zero = U@(W@np.zeros_like(rms))
                assert not np.any(zero)
                fits += 1
    by_index = {s["index"]: s for s in directions}; max_violation = 0.
    table = rows(ROOT / "A18_DEFECT_ORACLE.csv")
    for row in table:
        index = int(row["direction_index"]); s = by_index[index]; a = arrays[index]
        fold = int(row["fold"]); part = row["partition"]; base = "U" if row["target"] == "direct_defect" else "A"
        assert s["object_id"] == split_map[fold][part+"_object"]
        assert s["object_id"] not in split_map[fold]["train_objects"]
        if row["model"] == "UNCORRECTED_REDUCED":
            correction = np.zeros_like(a["zF"])
        else:
            U, W, rms = fitted[fold, base, int(row["requested_rank"])]
            correction = U@(U.T@a["defect"+base]) if row["model"] == "POD_EXACT_LATENT_ORACLE" else U@(W@(a["source_sketch"]/rms))
        z = a["z"+base]+correction; error = float(la.norm(a["zF"]-z))
        gain = float(-a["u"]@z-.5*(z@z)+s["material_gain_term"])
        m = gain-s["tau"]; v = float(la.norm(a["u"]+z))
        eps = 2*m/(np.sqrt(v*v+2*m)+v) if m > 0 else None
        bound = v*error+.5*error**2; violation = max(0., abs(s["Q_true"]-gain)-bound)
        for field, value in (("corrected_Qhat", gain), ("actual_margin", m), ("response_error_norm", error),
                             ("v", v), ("gain_error_bound", bound), ("gain_error_absolute", abs(s["Q_true"]-gain))):
            close(value, row[field], "B "+field)
        assert row["oracle_can_skip"] == str(eps is not None and error < eps)
        if eps is None:
            assert row["epsilon_crit"] == ""
        else:
            close(eps, row["epsilon_crit"], "critical radius")
            close(v*eps+.5*eps*eps, m, "critical radius equation", 1e-16)
        max_violation = max(max_violation, violation)
    summary = rows(ROOT / "results/A18_B_OBJECT_SUMMARY.csv")
    for saved in summary:
        rr = [r for r in table if (r["fold"], r["partition"], r["target"], r["model"], r["requested_rank"]) ==
              (saved["fold"], saved["partition"], saved["target"], saved["model"], saved["rank"])]
        if saved["method"] != "ALL_UNIQUE_MATCHED":
            rr = [r for r in rr if saved["method"] in json.loads(r["methods"])]
        n = len(rr); q = len({r["query_id"] for r in rr}); skip = sum(r["oracle_can_skip"] == "True" for r in rr)
        assert (n, q, skip) == (int(saved["unique_directions"]), int(saved["queries"]), int(saved["skip_count"]))
        close(skip/n, saved["skip_fraction"], "B skip fraction")
        close((1+q/n)/2, saved["ideal_skip_fraction_for_total_RHS_halving"], "B halving floor")
        close((q+n-skip)/(q+n), saved["counterfactual_RHS_ratio_with_init"], "B initialized cost")
    assert max_violation < 1e-12
    return {"saved_legal_directions_checked": len(directions), "complete_Gaussian_directions": len(cohort),
            "training_only_POD_ridge_refits": fits, "oracle_rows_recomputed": len(table),
            "matched_summary_rows_checked": len(summary), "max_gain_bound_violation": max_violation,
            "source_nullspace_oracle_recomputed": False,
            "scope": "Refits on saved legal measurement labels. Upstream fullJ/nullspace and original source hashes require their upstream/original files."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for fresh algebra results and replay receipt.")
    parser.add_argument("--hashes-only", action="store_true")
    args = parser.parse_args(); start = time.process_time(); checked = verify_files()
    if args.hashes_only:
        print(json.dumps(checked)); return
    assert args.output is not None, "Supply --output with a new directory."
    out = args.output.resolve()
    assert not out.exists(), "Output must be new; archived evidence is never overwritten."
    out.mkdir(parents=True)
    for sub in ("vendor", "verification"):
        (out / sub).mkdir()
    shutil.copytree(ROOT / "vendor", out / "vendor", dirs_exist_ok=True)
    shutil.copyfile(ROOT / "stage0_config.json", out / "stage0_config.json")
    shutil.copyfile(ROOT / "verification/verify_stage0_algebra.py", out / "verification/verify_stage0_algebra.py")
    subprocess.run([sys.executable, "-B", str(out / "verification/verify_stage0_algebra.py")], check=True, capture_output=True)
    algebra = json.loads((out / "verification/algebra_results.json").read_text())
    assert algebra["status"] == "PASS" and algebra["cases"] == 128
    result = {"status": "PASS", "files": checked, "A": audit_A(), "B": audit_B(),
              "algebra": {k: algebra[k] for k in ("status", "cases", "max_errors", "implementation")},
              "cpu_s": time.process_time()-start, "new_full_wave_rhs": 0, "NN_training_epochs": 0,
              "new_nonlinear_trajectories": 0,
              "scope": "PUBLIC_SAVED_LABEL_DIAGNOSTIC_REPLAY; NOT A NEW PHYSICAL EXPERIMENT OR FULL ORIGINAL-SOURCE REPLAY",
              "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "rss_units": "bytes on macOS, KiB on Linux; historical Stage0 budget receipts remain separate"}
    (out / "PUBLIC_REPLAY_VALIDATION.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
