# A18 Stage 0 frozen protocol

Owner: current Codex default; single bounded CPU worker. All upstream paths read-only. The source commit and analytic defaults are frozen in stage0_config.json before outcome calculation.

Full-wave RHS=0; NN epochs=0; nonlinear trajectories=0. Computation has a 3600 CPU-second and 4 GiB process RSS limit. Public projected current operators are reduced-model inputs; fullJ/fullJV/Js are evaluator-only. No new provider profile, kernel, remote job, training, push or automatic next stage.

Missing original A18 verifier/config/results remain missing. Rewritten verification is independently authored and does not inherit the reported original PASS. References were extracted verbatim from the supplied combined document.

After compaction/recovery read inputs/A18_CODEX_PROMPT.md, stage0_config.json and STATUS.json before work.

数值并列合同：首次运行前固定按 query score 最大绝对值的 1e-12 量化并列；B3 utility 保留 12 位小数，并列按原始 atom ID。用于避免复相位/存储置换下浮点舍入改变动作，不据结果调参。
