# Publication provenance and boundaries

Published on the owner's explicit `publish` instruction. Stage 0 research remains complete and stopped. The publication step did not add physics, training or scientific scenes.

`PUBLIC_MANIFEST.json` is the final public byte manifest (self excluded). `provenance/SOURCE_TO_PUBLIC_MANIFEST.json` pairs each included sealed source hash with its public output hash and transformation mode. `provenance/SEALED_DELIVERY_MANIFEST.json` is the verbatim original 977-file delivery manifest; it records the pre-publication local archive, not a list to rehash against projected public files.

Only absolute local path metadata was changed, through structured JSON/JSONL or per-cell CSV projection where applicable. Namespaces are specified in `provenance/PUBLIC_TRANSFORMATIONS.json`. These are provenance labels, not filesystem paths. All numeric fields, physical IDs, executed code, NPZ arrays and the three TeX/PDF pairs are unchanged. Legacy `INPUT_HASHES.json`, inventory SHA cells, query teacher hashes and freeze receipts retain original source hashes; public-current bytes use the public manifest. These two claims are distinct.

The omitted duplicate independent replay tree and transient LaTeX `.aux/.out/.toc` bookkeeping are individually listed with original hashes in `provenance/PUBLIC_OMISSIONS.json`. All failures, failed numerical-attempt partial outputs, logs, ledgers, historical resource receipts, code and 28 reviewed page images are retained. Virtual environments, caches, credentials, external private physics kernels and third-party paper PDFs were never part of the sealed delivery and are not added here.

The pinned A17 repository supplies upstream provenance. This archive does not redistribute missing originals or assume rights to third-party papers. Vendor code remains byte-exact and attributed to that pinned repository; its upstream terms apply. No new license is asserted for outside materials.

Original scholarly citations/access records are in `inputs/REFERENCES.md`; the fresh Stage 0 metadata/access ledger is in `literature/`. Inherited access dates and HTTP429 failures are preserved. No claim of exhaustive novelty, learned-model superiority, a deterministic Maxwell certificate, measured speedup or TAP acceptance is introduced by publication.
