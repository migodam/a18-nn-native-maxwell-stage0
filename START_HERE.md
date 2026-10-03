# A18 NN Native Maxwell — Stage 0 public research archive

Stage 0 is complete and stopped. This archive contains three manuscripts (English study, Chinese report to Professor Chen, and full Chinese explanation), the nine specified audits, code, derived arrays, figures, provenance, costs and failures.

**A: `A_GO_TINY_PILOT`**, a recommendation for a future bounded candidate-generator experiment. **B: `B_DATA_INCOMPLETE`** over four objects; the complete Gaussian diagnostic is **`B_NO_MARGIN_OR_COST_HEADROOM_FOR_TESTED_POD_BASES`**. The archive contains no trained NN: NN epochs, new full-wave RHS, new nonlinear trajectories and GPU training are all **zero**. `FUTURE_TRAINING_CONFIG.json` remains disabled.

## Read this study

Start with the [decision](https://raw.githubusercontent.com/migodam/a18-nn-native-maxwell-stage0/main/A18_STAGE0_DECISION.md) and [single-file analysis bundle](https://raw.githubusercontent.com/migodam/a18-nn-native-maxwell-stage0/main/ANALYSIS_BUNDLE.md). The bundle includes the common claim/reference/failure ledgers and text extracted from all three final PDFs; equations and layout in the PDFs/TeX remain authoritative.

| Manuscript | PDF | Standalone editable source |
|---|---|---|
| English Stage 0 research, 8 pages | [PDF](manuscript/STAGE0_RESEARCH_EN.pdf) | [TeX](https://raw.githubusercontent.com/migodam/a18-nn-native-maxwell-stage0/main/manuscript/STAGE0_RESEARCH_EN.tex) |
| 给陈教授的中文汇报, 6 pages | [PDF](manuscript/REPORT_TO_PROF_CHEN_ZH.pdf) | [TeX](https://raw.githubusercontent.com/migodam/a18-nn-native-maxwell-stage0/main/manuscript/REPORT_TO_PROF_CHEN_ZH.tex) |
| 中文完整讲解, 14 pages | [PDF](manuscript/COMPLETE_EXPLANATION_ZH.pdf) | [TeX](https://raw.githubusercontent.com/migodam/a18-nn-native-maxwell-stage0/main/manuscript/COMPLETE_EXPLANATION_ZH.tex) |

The PDF and TeX bytes are identical to the reviewed local delivery. Native LaTeX compilation succeeded; all 28 exported PDF pages were visually inspected. See [QA evidence](qa/PDF_QA_RESULTS.json).

## Findings and scope

The complete audit covers **36 receiver queries, 47,712 unique one-swap moves, 43,108 feasible moves and 5,208 incoming labels**, on four objects and twelve frozen states. Moves are correlated. The original family-prefix coverage is reproduced. B2 O/P/M union improves near-contact Cov12 from 0.3528 to 0.5178, below 0.70. B3 mean Cov12 is 0.6259. None of the frozen cheap baselines passes all coverage gates; this leaves an opportunity to test candidate generation, rather than demonstrating NN success.

B has 204 complete Gaussian matched measurement directions plus one isolated legal voxel direction. 194 voxel matched vectors and all full-current labels are missing. The primary prescribed test is Fold 1: train on smooth, test on near-contact. Smooth in Fold 3 is validation, not a second test. For the 18 primary B4 finalists, anchor-tail POD ranks 8/16/32 each permit 10/18 hypothetical skips, exactly the uncorrected anchor count; direct POD permits 0/18. 55.6% misses the 60% opportunity gate and this matched cohort's 75% initialized RHS-halving requirement. No physical solve was skipped and no runtime acceleration was measured.

Source-sketch diagnostics use the saved Gaussian material gauge and exact evaluation operators; they do not establish an online predictor or unresolved-current labels. The negative B result is limited to the tested POD/ridge pipeline. Statistical safety calibration and a global novelty claim remain open; literature access failures are retained.

## Evidence navigation

The nine requested files are `A18_DATA_INVENTORY.csv`, `A18_QUERY_MANIFEST.jsonl`, `A18_SPLITS.json`, `A18_FEATURE_COSTS.csv`, `A18_ANALYTIC_SHORTLIST_BASELINES.csv`, `A18_DEFECT_LABEL_AVAILABILITY.csv`, `A18_DEFECT_ORACLE.csv`, `A18_COST_LEDGER.jsonl`, and `A18_STAGE0_DECISION.md`. See the [file index](FILE_INDEX.md), [claim ledger](CLAIM_LEDGER.csv), [failure ledger](FAILURE_LEDGER.csv), [references](literature/REFERENCE_LEDGER.csv), and [figure guide](figures/README.md).

Fixed upstream: [A17 at `56097653e6a814399581c3759fc3e3e8f0c2284c`](https://github.com/migodam/a17-maxwell-certified-exchange/tree/56097653e6a814399581c3759fc3e3e8f0c2284c). Original/public manifests and original teacher files were separately hashed during the local study; the historical [source-integrity receipt](verification/FINAL_SOURCE_INTEGRITY.json) is preserved with path metadata projected. The original A18 validation/config/results were absent; the new algebra checks are equivalent implementations, not an original-script reproduction.

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
