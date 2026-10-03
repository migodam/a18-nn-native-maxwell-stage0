"""Build a curated public copy from the sealed A18 delivery; never alter originals."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def seal_public(dest):
    excluded = {"PUBLIC_MANIFEST.json"}
    items = [{"path": str(p.relative_to(dest)), "sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(dest.rglob("*")) if p.is_file() and ".git" not in p.parts
             and str(p.relative_to(dest)) not in excluded and not p.name.startswith(".DS_Store")
             and "__pycache__" not in p.parts and "public_reproductions" not in p.parts]
    write_json(dest / "PUBLIC_MANIFEST.json", {
        "schema": "a18.public.snapshot.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "56097653e6a814399581c3759fc3e3e8f0c2284c",
        "excludes": ["self", "Git metadata", "OS metadata", "Python bytecode", "fresh public_reproductions"],
        "hash_scope": "Bytes in this public copy. Historical original hashes are in provenance and the original receipts.",
        "files": items})
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tools-dir", required=True, type=Path)
    parser.add_argument("--repo", default="migodam/a18-nn-native-maxwell-stage0")
    args = parser.parse_args()
    source = args.source_dir.resolve(); dest = args.output.resolve()
    assert not dest.exists(), "Output must be new."
    sealed = json.loads((source / "DELIVERY_MANIFEST.json").read_text())
    cfg = json.loads((source / "stage0_config.json").read_text())
    for item in sealed["files"]:
        assert sha(source / item["path"]) == item["sha256"], ("sealed delivery changed", item["path"])
    mapping = sorted([(cfg["source_root"], "upstream-a17://"),
                      (cfg["original_root"], "original-a17://"),
                      (str(source), "a18://"),
                      (str(source.parent / "Gaussian/A18"), "original-a18://"),
                      (str(source.parent), "workspace://")], key=lambda x:len(x[0]), reverse=True)
    transforms = []; omissions = []; included = []
    def text_project(value):
        for old, new in mapping:
            value = value.replace(old+"/", new).replace(old, new)
        return value
    def project(value):
        if isinstance(value, str):
            return text_project(value)
        if isinstance(value, list):
            return [project(v) for v in value]
        if isinstance(value, dict):
            return {text_project(k): project(v) for k, v in value.items()}
        return value
    dest.mkdir(parents=True); (dest / "provenance").mkdir()
    for item in sealed["files"]:
        rel = item["path"]; path = source / rel
        if rel.startswith("reproductions/"):
            omissions.append({**item, "reason": "Duplicate independent run tree; comparison, scope, counts and hashes retained in verification/REPRODUCTION_VALIDATION.json."})
            continue
        if path.suffix in {".aux", ".out", ".toc"}:
            omissions.append({**item, "reason": "LaTeX transient compilation bookkeeping; final sources, PDFs, logs and all page QA retained."})
            continue
        public_rel = "provenance/LOCAL_START_HERE.md" if rel == "START_HERE.md" else rel
        out = dest / public_rel; out.parent.mkdir(parents=True, exist_ok=True)
        mode = "VERBATIM"
        if path.suffix in {".json", ".jsonl", ".csv", ".md", ".log", ".txt", ".tex", ".py", ".svg"}:
            original = path.read_text(); modified = text_project(original)
            if modified != original:
                assert path.suffix not in {".py", ".tex", ".svg"}, "Executable/scientific sources cannot be rewritten implicitly."
                mode = "PATH_METADATA_PROJECTION"
                if path.suffix == ".json":
                    obj = json.loads(original); updated = project(obj); write_json(out, updated)
                    assert updated == project(obj)
                elif path.suffix == ".jsonl":
                    records = [json.loads(line) for line in original.splitlines()]
                    updated = [project(r) for r in records]
                    out.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in updated))
                    assert updated == project(records)
                elif path.suffix == ".csv":
                    with path.open(newline="") as f:
                        reader = csv.DictReader(f); fields = reader.fieldnames; records = list(reader)
                    updated = [{key:text_project(value) for key, value in row.items()} for row in records]
                    with out.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(updated)
                    # Every cell can change only by one of the declared metadata path substitutions.
                    assert all(new == {k:text_project(v) for k,v in old.items()} for old,new in zip(records,updated))
                else:
                    out.write_text(modified)
            else:
                shutil.copyfile(path, out)
        else:
            shutil.copyfile(path, out)
        record = {"source_path": rel, "public_path": public_rel, "source_sha256": item["sha256"],
                  "output_sha256": sha(out), "source_bytes": item["bytes"], "output_bytes": out.stat().st_size,
                  "mode": mode}
        if mode == "VERBATIM":
            assert record["source_sha256"] == record["output_sha256"]
        else:
            transforms.append(record)
        included.append(record)
    shutil.copyfile(source / "DELIVERY_MANIFEST.json", dest / "provenance/SEALED_DELIVERY_MANIFEST.json")
    write_json(dest / "provenance/SOURCE_TO_PUBLIC_MANIFEST.json", included)
    write_json(dest / "provenance/PUBLIC_TRANSFORMATIONS.json", {
        "rule": "Only declared local path metadata was projected. Numeric fields, atom IDs, arrays, executable code and three manuscript sources/PDFs are unchanged.",
        "namespaces": {"a18://": "Original sealed local A18 delivery", "upstream-a17://": "Pinned public A17 snapshot",
                       "original-a17://": "Original A17 inputs; unavailable originals are not inferred", "original-a18://": "Original supplied A18 documents",
                       "workspace://": "Other paths relative to the original workspace"},
        "original_frozen_configuration_sha256": sha(source / "stage0_config.json"),
        "original_sealed_delivery_manifest_sha256": sha(source / "DELIVERY_MANIFEST.json"),
        "public_configuration_sha256": sha(dest / "stage0_config.json"), "changed_files": transforms})
    write_json(dest / "provenance/PUBLIC_OMISSIONS.json", omissions)
    (dest / "tools").mkdir()
    for tool in args.tools_dir.glob("*.py"):
        shutil.copyfile(tool, dest / "tools" / tool.name)
    shutil.copyfile(Path(__file__), dest / "tools/build_public_snapshot.py")
    (dest / "requirements.txt").write_text("numpy==2.5.3\nscipy==1.18.1\nmatplotlib==3.11.1\n")
    (dest / ".gitignore").write_text("__pycache__/\n*.pyc\n.DS_Store\n.venv/\npublic_reproductions/\n*.aux\n*.out\n*.toc\n")
    raw = "https://raw.githubusercontent.com/"+args.repo+"/main/"
    upstream = "https://github.com/migodam/a17-maxwell-certified-exchange/tree/"+cfg["source_commit"]
    entry = f"""# A18 NN Native Maxwell — Stage 0 public research archive

Stage 0 is complete and stopped. This archive contains three manuscripts (English study, Chinese report to Professor Chen, and full Chinese explanation), the nine specified audits, code, derived arrays, figures, provenance, costs and failures.

**A: `A_GO_TINY_PILOT`**, a recommendation for a future bounded candidate-generator experiment. **B: `B_DATA_INCOMPLETE`** over four objects; the complete Gaussian diagnostic is **`B_NO_MARGIN_OR_COST_HEADROOM_FOR_TESTED_POD_BASES`**. The archive contains no trained NN: NN epochs, new full-wave RHS, new nonlinear trajectories and GPU training are all **zero**. `FUTURE_TRAINING_CONFIG.json` remains disabled.

## Read this study

Start with the [decision]({raw}A18_STAGE0_DECISION.md) and [single-file analysis bundle]({raw}ANALYSIS_BUNDLE.md). The bundle includes the common claim/reference/failure ledgers and text extracted from all three final PDFs; equations and layout in the PDFs/TeX remain authoritative.

| Manuscript | PDF | Standalone editable source |
|---|---|---|
| English Stage 0 research, 8 pages | [PDF](manuscript/STAGE0_RESEARCH_EN.pdf) | [TeX]({raw}manuscript/STAGE0_RESEARCH_EN.tex) |
| 给陈教授的中文汇报, 6 pages | [PDF](manuscript/REPORT_TO_PROF_CHEN_ZH.pdf) | [TeX]({raw}manuscript/REPORT_TO_PROF_CHEN_ZH.tex) |
| 中文完整讲解, 14 pages | [PDF](manuscript/COMPLETE_EXPLANATION_ZH.pdf) | [TeX]({raw}manuscript/COMPLETE_EXPLANATION_ZH.tex) |

The PDF and TeX bytes are identical to the reviewed local delivery. Native LaTeX compilation succeeded; all 28 exported PDF pages were visually inspected. See [QA evidence](qa/PDF_QA_RESULTS.json).

## Findings and scope

The complete audit covers **36 receiver queries, 47,712 unique one-swap moves, 43,108 feasible moves and 5,208 incoming labels**, on four objects and twelve frozen states. Moves are correlated. The original family-prefix coverage is reproduced. B2 O/P/M union improves near-contact Cov12 from 0.3528 to 0.5178, below 0.70. B3 mean Cov12 is 0.6259. None of the frozen cheap baselines passes all coverage gates; this leaves an opportunity to test candidate generation, rather than demonstrating NN success.

B has 204 complete Gaussian matched measurement directions plus one isolated legal voxel direction. 194 voxel matched vectors and all full-current labels are missing. The primary prescribed test is Fold 1: train on smooth, test on near-contact. Smooth in Fold 3 is validation, not a second test. For the 18 primary B4 finalists, anchor-tail POD ranks 8/16/32 each permit 10/18 hypothetical skips, exactly the uncorrected anchor count; direct POD permits 0/18. 55.6% misses the 60% opportunity gate and this matched cohort's 75% initialized RHS-halving requirement. No physical solve was skipped and no runtime acceleration was measured.

Source-sketch diagnostics use the saved Gaussian material gauge and exact evaluation operators; they do not establish an online predictor or unresolved-current labels. The negative B result is limited to the tested POD/ridge pipeline. Statistical safety calibration and a global novelty claim remain open; literature access failures are retained.

## Evidence navigation

The nine requested files are `A18_DATA_INVENTORY.csv`, `A18_QUERY_MANIFEST.jsonl`, `A18_SPLITS.json`, `A18_FEATURE_COSTS.csv`, `A18_ANALYTIC_SHORTLIST_BASELINES.csv`, `A18_DEFECT_LABEL_AVAILABILITY.csv`, `A18_DEFECT_ORACLE.csv`, `A18_COST_LEDGER.jsonl`, and `A18_STAGE0_DECISION.md`. See the [file index](FILE_INDEX.md), [claim ledger](CLAIM_LEDGER.csv), [failure ledger](FAILURE_LEDGER.csv), [references](literature/REFERENCE_LEDGER.csv), and [figure guide](figures/README.md).

Fixed upstream: [A17 at `{cfg['source_commit']}`]({upstream}). Original/public manifests and original teacher files were separately hashed during the local study; the historical [source-integrity receipt](verification/FINAL_SOURCE_INTEGRITY.json) is preserved with path metadata projected. The original A18 validation/config/results were absent; the new algebra checks are equivalent implementations, not an original-script reproduction.

The historical independent local replay matched the numerical tables and 295 derived arrays exactly, with timing excluded. Its [comparison receipt](verification/REPRODUCTION_VALIDATION.json) is included; the duplicate replay tree is omitted with original hashes recorded. Historical metered numerical CPU was 49.1 seconds; all metered child processes including authoring/failures were 80.3 seconds, peak RSS 0.824 GiB. Native UI compiler CPU was not instrumented. These are research receipts; publication utilities are accounted separately.

## Reproduce and inspect

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The portable public adapter rehashes this archive, recomputes A metrics from complete saved incoming teachers and recorded shortlists, refits train-only POD/ridge models on saved legal Gaussian labels, recomputes all 2,856 margin-oracle rows and 252 B summary rows, and runs 128 algebra cases. It does not regenerate upstream labels, re-evaluate endpoint rankings or reverify unavailable original files.

```sh
python -B tools/public_audit.py --hashes-only
python -B tools/public_audit.py --output public_reproductions/check1
python -B tools/replot.py --output public_reproductions/figures1
```

`PUBLIC_MANIFEST.json` hashes the public copy. `provenance/SOURCE_TO_PUBLIC_MANIFEST.json` records both original and public file hashes. Local paths are replaced with declared provenance namespaces; numeric scientific values, executable code, all arrays and manuscripts are preserved. Historical input/config hashes refer to the original sealed delivery, rather than the projected public metadata. [Publication notes](PUBLICATION_NOTES.md) describe this distinction and omissions.

Repository publication is not a new research stage. Candidate Generator first remains the next hypothesis to test; B training is not recommended from this evidence. Stage 0 remains stopped.
"""
    (dest / "START_HERE.md").write_text(entry)
    (dest / "README.md").write_text(f"""# A18 NN Native Maxwell — Stage 0

[Read the study](START_HERE.md) · [Web GPT raw entry]({raw}START_HERE.md) · [Analysis bundle](ANALYSIS_BUNDLE.md)

Completed feasibility study using the fixed A17 archive. **A_GO_TINY_PILOT** is a future experiment recommendation. **B_DATA_INCOMPLETE** remains separate from the Gaussian finding **B_NO_MARGIN_OR_COST_HEADROOM_FOR_TESTED_POD_BASES**. No NN training, new full-wave RHS or nonlinear trajectory was run.

- [English study](manuscript/STAGE0_RESEARCH_EN.pdf) / [TeX](manuscript/STAGE0_RESEARCH_EN.tex)
- [给陈教授的汇报](manuscript/REPORT_TO_PROF_CHEN_ZH.pdf) / [TeX](manuscript/REPORT_TO_PROF_CHEN_ZH.tex)
- [中文完整讲解](manuscript/COMPLETE_EXPLANATION_ZH.pdf) / [TeX](manuscript/COMPLETE_EXPLANATION_ZH.tex)
- [Nine audits and decision](A18_STAGE0_DECISION.md), [claims](CLAIM_LEDGER.csv), [failures](FAILURE_LEDGER.csv), [provenance](PUBLICATION_NOTES.md), [reproduction](REPRODUCIBILITY.md).

Public saved-label replay and replotting are provided; original full Stage 0 replay requires the pinned upstream and unavailable original local files described in the reproduction notes. The three reviewed TeX/PDF pairs and all numerical arrays are byte-preserved.
""")
    (dest / "REPRODUCIBILITY.md").write_text("""# Reproduction scopes

## Portable public saved-label replay

Tested publication runtime: Python 3.13.13, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.1, Apple Silicon CPU. `requirements.txt` records these observed versions. No installation was performed during publication. Use a suitable Python environment and run from the repository root:

```sh
python -B tools/public_audit.py --hashes-only
python -B tools/public_audit.py --output public_reproductions/check1
python -B tools/replot.py --output public_reproductions/figures1
```

Choose new output directories. They are ignored by Git, and archived evidence is not overwritten. `tools/public_audit.py` reports fresh results in its output directory. The committed publication replay receipt records the tested code and input hashes in `verification/PUBLIC_REPLAY_VALIDATION.json`. That historical receipt may refer to a preliminary manifest before the receipt itself was added; the current manifest hashes the final receipt and tool bytes.

The adapter verifies the public manifest, full per-query incoming-table coverage and archived shortlist utility/recall/regret metrics, object Cov12 summaries, 205 direction-label identities and gains, all 18 train-only POD/ridge fits (with singular-vector gauge-independent comparison), 2,856 oracle rows and 252 summary rows, and 128 equivalent non-Hermitian algebra cases. It uses saved labels, including exact evaluator u, only in evaluation. It does not test a trained neural architecture. Replotting regenerates four PNG/SVG pairs from saved CSVs; pixel-identical rendering is not promised.

## Original full Stage 0 replay, archived code

`code/`, `vendor/`, and `verification/*.py` retain their original bytes. The full `code/reproduce.py` entry requires a Git checkout of the pinned A17 public repository plus original A17/A18 inputs, including raw teacher metadata for original-hash checks. It also assumes the original macOS resource-unit conventions and optional PDF tool paths. This dependency is explicit: the public adapter does not silently replace originals or report unavailable original-hash checks as a fresh PASS.

Public `stage0_config.json` has the same frozen scientific parameters but projected source-root metadata. For full original replay, bind its two source roots to real local locations in a new scratch copy, and resolve original document locations as described by the archived source-audit script. This changes configuration metadata bytes and must be recorded. Original frozen config SHA and original input hashes remain historical provenance. The new public adapter needs no such binding.

`verification/REPRODUCTION_VALIDATION.json` records the already-completed independent local full replay: exact scientific tables/arrays, timing excluded. Its duplicate output tree is omitted from publication; every omitted file's original hash is in `provenance/PUBLIC_OMISSIONS.json`.

The portable adapter does not rerun: public F1 feature construction/permutation/phase canaries requiring projected upstream workspaces, endpoint reconstruction/Qhat ranking, source-nullspace fullJ diagnostics, upstream dataset generation, historical timing, unavailable original provenance, or PDF compiler/visual acceptance. Their original receipts, code, costs, failures and scope remain available. None of these modes runs new full-wave RHS, NN epochs or nonlinear trajectories.

## Manuscripts and figures

Each `.tex` is standalone with embedded TikZ/pgfplots content. The English document uses ordinary LaTeX; Chinese documents use `ctex` with Fandol fonts. The completed local native-editor and Tectonic compilation receipts plus all 28 page images remain in `qa/`. Recompilation may change PDF timestamps and rendering; the published reviewed PDFs are byte-preserved, rather than newly certified by compilation during publication.
""")
    (dest / "PUBLICATION_NOTES.md").write_text("""# Publication provenance and boundaries

Published on the owner's explicit `publish` instruction. Stage 0 research remains complete and stopped. The publication step did not add physics, training or scientific scenes.

`PUBLIC_MANIFEST.json` is the final public byte manifest (self excluded). `provenance/SOURCE_TO_PUBLIC_MANIFEST.json` pairs each included sealed source hash with its public output hash and transformation mode. `provenance/SEALED_DELIVERY_MANIFEST.json` is the verbatim original 977-file delivery manifest; it records the pre-publication local archive, not a list to rehash against projected public files.

Only absolute local path metadata was changed, through structured JSON/JSONL or per-cell CSV projection where applicable. Namespaces are specified in `provenance/PUBLIC_TRANSFORMATIONS.json`. These are provenance labels, not filesystem paths. All numeric fields, physical IDs, executed code, NPZ arrays and the three TeX/PDF pairs are unchanged. Legacy `INPUT_HASHES.json`, inventory SHA cells, query teacher hashes and freeze receipts retain original source hashes; public-current bytes use the public manifest. These two claims are distinct.

The omitted duplicate independent replay tree and transient LaTeX `.aux/.out/.toc` bookkeeping are individually listed with original hashes in `provenance/PUBLIC_OMISSIONS.json`. All failures, failed numerical-attempt partial outputs, logs, ledgers, historical resource receipts, code and 28 reviewed page images are retained. Virtual environments, caches, credentials, external private physics kernels and third-party paper PDFs were never part of the sealed delivery and are not added here.

The pinned A17 repository supplies upstream provenance. This archive does not redistribute missing originals or assume rights to third-party papers. Vendor code remains byte-exact and attributed to that pinned repository; its upstream terms apply. No new license is asserted for outside materials.

Original scholarly citations/access records are in `inputs/REFERENCES.md`; the fresh Stage 0 metadata/access ledger is in `literature/`. Inherited access dates and HTTP429 failures are preserved. No claim of exhaustive novelty, learned-model superiority, a deterministic Maxwell certificate, measured speedup or TAP acceptance is introduced by publication.
""")
    file_index = "# File index\n\nA18 Stage 0 public archive. Scientific scope and reproduction limitations: [START_HERE.md](START_HERE.md).\n\n"
    for folder, label in [("manuscript", "Three manuscripts, source and PDF"), ("results", "Derived labels, models, summaries and execution evidence"),
                          ("verification", "Algebra, interfaces, source provenance and replay receipts"), ("figures", "Replot-capable figure data and exports"),
                          ("inputs", "Supplied research documents, restored references"), ("literature", "Fresh scholarly evidence and access failures"),
                          ("qa", "Native compiler, export, text extraction and 28 page visual receipts"),
                          ("code", "Original executed Stage 0 implementation"), ("vendor", "Byte-exact pinned upstream CPU implementation"),
                          ("tools", "Public saved-label replay, replot and snapshot builder"), ("provenance", "Source/public hashes, projections and omissions")]:
        file_index += f"- [{folder}/]({folder}/): {label}.\n"
    file_index += "\nFull per-file size/hash listing: [PUBLIC_MANIFEST.json](PUBLIC_MANIFEST.json). The original required nine audit files are at the repository root.\n"
    (dest / "FILE_INDEX.md").write_text(file_index)
    bundle = "# A18 Stage 0 — web reading bundle\n\nRead [START_HERE.md](START_HERE.md) for scope and navigation. This combines decision, common ledgers and extracted text from all three final PDFs. PDF/TeX equations and tables are authoritative when text extraction loses formatting. No new scientific result is introduced here.\n\n"
    bundle += "## Branch decision\n\n"+(dest / "A18_STAGE0_DECISION.md").read_text()+"\n"
    for rel, title in [("CLAIM_LEDGER.csv", "Claim ledger"), ("FAILURE_LEDGER.csv", "Failure ledger"), ("literature/REFERENCE_LEDGER.csv", "Reference ledger")]:
        bundle += f"\n## {title}\n\n```csv\n"+(dest / rel).read_text()+"```\n"
    for stem, title in [("STAGE0_RESEARCH_EN", "English study"), ("REPORT_TO_PROF_CHEN_ZH", "给陈教授的中文汇报"), ("COMPLETE_EXPLANATION_ZH", "中文完整讲解")]:
        bundle += f"\n## {title} — extracted PDF text\n\n"+(dest / "qa" / (stem+"_extracted.txt")).read_text()+"\n"
    (dest / "ANALYSIS_BUNDLE.md").write_text(bundle)
    count = seal_public(dest)
    print(json.dumps({"status":"BUILT", "public_files":count, "sealed_source_files_checked":len(sealed["files"]),
                      "included_source_files":len(included), "path_projected_files":len(transforms), "omitted_files":len(omissions),
                      "originals_unchanged":True, "scientific_values_unchanged":True}))


if __name__ == "__main__":
    main()
