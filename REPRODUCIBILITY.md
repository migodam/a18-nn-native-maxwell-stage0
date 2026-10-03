# Reproduction scopes

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
