# A18 sources and access registry

Access date: 2026-10-02. A17 repository reads are pinned to commit `56097653e6a814399581c3759fc3e3e8f0c2284c`. Reports and source code were inspected through the GitHub connector. This package does not claim a raw-array reanalysis of the full A17 archive. No copyrighted paper PDFs are redistributed.

## Project sources

Root: https://github.com/migodam/a17-maxwell-certified-exchange/tree/56097653e6a814399581c3759fc3e3e8f0c2284c

- **[A17-start]** `START_HERE.md`: measured scope, limitations, read order, external kernels.
- **[A17-decision]** `A17_DECISION.md`: local design GO, score/coverage separation, learned-priority NO-GO, certificates unavailable.
- **[A17-report]** `A17_FINAL_RESEARCH_REPORT_ZH.md`: object-wise ratios, short-pool coverage, illumination count, material dimensions, 1164 finalist RHS, weak-space scope.
- **[A17-NN]** `A17_NN_PILOT.md`: 47712 rows/43108 feasible labels, object split, six runs, cost exclusions.
- **[A17-cost]** `A17_COST_AUDIT.md`: matched costs, initialization/setup distinctions, no demonstrated net acceleration.
- **[A17-core]** `A17_R1/code/exchange_core.py`: realification, Galerkin/Schur formulas, legal stability checks, finite directional gain.
- **[A17-run]** `A17_R1/code/run_exchange.py`: `balanced_ids`, `moves`, `evaluate_neighborhood`, `policy_run`, and `run`; scalar label storage, accepted endpoint arrays, fullJ versus projected fullJV storage.
- **[A17-state]** `A17_R1/code/maxwell_state.py`: `public_workspace`, `EvaluatorContext.offline_gaussian_J`, `ProbeContext`, `TeacherContext`; separation of projected public maps and full actions.

To form an exact file URL, append the listed path to `https://github.com/migodam/a17-maxwell-certified-exchange/blob/56097653e6a814399581c3759fc3e3e8f0c2284c/`.

## Primary research and method sources

**[R1]** Tiantian Yin, Kai Tan, Xudong Chen. *Far-Field Approximation Learning Method for Millimeter-Wave Short-Range Imaging*. IEEE TAP, 2023. DOI: https://doi.org/10.1109/TAP.2023.3242424
Publication abstract: https://www.researchgate.net/publication/368409180_Far-Field_Approximation_Learning_Method_for_Millimeter-Wave_Short-Range_Imaging
Access: paper abstract retrieved; full text not obtained. Claims restricted to Fourier-based approximate input, correction of approximation/limited-aperture distortions, and reflectivity output. Exact layer/loss/training-size audit remains OPEN.

**[R2]** Yu Liu et al. *SOM-Net: Unrolling the Subspace-based Optimization for Solving Full-wave Inverse Scattering Problems*. IEEE TGRS, DOI 10.1109/TGRS.2022.3217162. Author preprint https://arxiv.org/abs/2209.03567 ; full PDF https://arxiv.org/pdf/2209.03567 . 2022.
Access: full PDF text, especially Algorithm 1 and equations (6)–(11), inspected. The method alternates learned current updates and explicit material updates. PDF screenshot service failed, so no claim of visual figure/table QA is made. The audit does not quote experimental tables.

**[R3]** Rui Guo, Zhichao Lin, Tao Shan, Xiaoqian Song, Maokun Li, Fan Yang, Shenheng Xu, Aria Abubakar. *Physics Embedded Deep Neural Network for Solving Full-Wave Inverse Scattering Problems*. IEEE TAP 70(8), 6148–6159, 2022; early online 2021. DOI: https://doi.org/10.1109/TAP.2021.3102135
Author-uploaded full text: https://www.researchgate.net/publication/353782924_Physics_Embedded_Deep_Neural_Network_for_Solving_Full-wave_Inverse_Scattering_Problems
Access: author-uploaded full-text extraction, including VIE/FWD solver definitions and equations (11)–(16), inspected. Direct PDF download failed; no visual architecture-figure QA claimed. Do not confuse it with the related VIE-forward-only paper, DOI 10.1109/TAP.2021.3070152.

**[R4]** Huilin Cao, Haoran Sun, Jingwei Liang, Tao Shan, Yuxiang Ren, Maokun Li. *Physics-Data Hybrid Driven Edge-Featured Graph Attention Networks for Surface Currents Learning of 3D PEC Targets*. IEEE TAP, 2025. DOI: https://doi.org/10.1109/TAP.2025.3602467
Author publication list: https://yuxiangren.github.io/publications/
IEEE record: https://ieeexplore.ieee.org/document/11146424/
Publication abstract record: https://www.researchgate.net/publication/395174268_Physics-Data_Hybrid_Driven_Edge-Featured_Graph_Attention_Networks_for_Surface_Currents_Learning_of_3D_PEC_Targets
Access: title/authors/year and abstract-level method scope; full text unavailable in this session. Layer counts, discretization details, exact edge formula, split and speedup denominator remain OPEN. Do not reproduce reported speed percentages without that audit.

**[R5]** Zhun Wei, Xudong Chen. *Physics-Inspired Convolutional Neural Network for Solving Full-Wave Inverse Scattering Problems*. IEEE TAP 67(9), 6138–6148, 2019. DOI: https://doi.org/10.1109/TAP.2019.2922779
Publication abstract: https://www.researchgate.net/publication/333889082_Physics-Inspired_Convolutional_Neural_Network_for_Solving_Full-Wave_Inverse_Scattering_Problems
Access: primary paper abstract reproduced at publication record; full text not obtained. Abstract supports induced-current learning, bypass of major current and learning the smaller correction, and multi-label loss. No deeper layer/formula assertions.

**[R6]** Lianlin Li et al. *DeepNIS: Deep Neural Network for Nonlinear Electromagnetic Inverse Scattering*. IEEE TAP, 2019; author preprint 2018. https://arxiv.org/abs/1810.03990 ; https://arxiv.org/pdf/1810.03990
Access: full PDF text, BP-image input and cascaded complex CNN method paragraphs inspected. Screenshot service failed; numerical tables not used.

**[R7]** Sebastian Lunz, Andreas Hauptmann, Tanja Tarvainen, Carola-Bibiane Schonlieb, Simon Arridge. *On Learned Operator Correction in Inverse Problems*. SIAM J. Imaging Sciences 14(1), 92–127, 2021. DOI: https://doi.org/10.1137/20M1338460 ; author preprint https://arxiv.org/abs/2005.07069
Access: full-text sections 3–4 on correction and adjoint consistency inspected. Screenshot service failed. No empirical performance figures used.

**[R8]** Thomas O'Leary-Roseberry, Peng Chen, Umberto Villa, Omar Ghattas. *Derivative-Informed Neural Operator: An Efficient Framework for High-Dimensional Parametric Derivative Learning*. JCP 496, 112555, 2024. DOI: https://doi.org/10.1016/j.jcp.2023.112555 ; https://arxiv.org/abs/2206.10745 ; https://tomoleary.github.io/dino/
Access: primary abstract/project method description, not an equation-by-equation full-text audit.

**[R9]** Dingcheng Luo, Thomas O'Leary-Roseberry, Peng Chen, Omar Ghattas. *Efficient PDE-Constrained Optimization Under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators*. SIAM J. Scientific Computing 47(4), C899–C931, 2025. DOI: https://doi.org/10.1137/23M157956X
Access: publisher abstract/method scope; no detailed empirical reproduction.

**[R10]** Bernard Haasdonk, Hendrik Kleikamp, Mario Ohlberger, Felix Schindler, Tizian Wenzel. *A new certified hierarchical and adaptive RB-ML-ROM surrogate model for parametrized PDEs*. Author preprint 2022. https://arxiv.org/abs/2204.13454
Access: primary abstract and authors' conference description. Its rigorous a posteriori hierarchy is a prior-art boundary, not evidence that A18 inherits a certificate.

**[R11]** *Closure Learning for Nonlinear Model Reduction Using Deep Residual Neural Network*. Fluids 5(1), 39, 2020. DOI: https://doi.org/10.3390/fluids5010039
Access: primary journal abstract/method summary. Resolved/unresolved closure is established prior art.

**[R12]** Juho Lee et al. *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*. ICML 2019. https://proceedings.mlr.press/v97/lee19d.html
Access: primary paper landing page. A candidate proposal requires equivariant per-element outputs, not only an invariant pooled output.

**[R13]** Petar Velickovic et al. *Graph Attention Networks*. ICLR 2018. https://arxiv.org/abs/1710.10903
Access: primary abstract. Generic attention alone does not encode Maxwell phase or Schur coupling.

**[R14]** Anastasios N. Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. https://arxiv.org/abs/2107.07511
Access: primary method reference. The finite-order-statistic calculation in A18 is a direct mathematical application, not a new theorem or a demonstrated A18 guarantee.

**[R15]** Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster. *Conformal Risk Control*. ICLR 2024. https://research.google/pubs/conformal-risk-control/
Access: authors' primary abstract. Expected monotone-loss risk control is not automatically conditional wrong-acceptance control after an adaptive shortlist.

## Evidence classification

- PROJECT_REPORTED: metrics directly stated by pinned A17 reports.
- SOURCE_VERIFIED: implementation/data-saving behavior directly visible in pinned source.
- DERIVED: algebra or cost calculation in this package.
- PROPOSED: pilot sizes, thresholds and architecture options; not achieved results.
- OPEN: inaccessible full-text detail or uninspected archive arrays.

A source access failure is not evidence that a paper lacks a feature. Novelty judgments below are provisional where full text is OPEN. No first-ever claim is made.
**[R16]** Lianghao Cao, Thomas O'Leary-Roseberry, Prashant K. Jha, J. Tinsley Oden, Omar Ghattas. *Residual-based error correction for neural operator accelerated infinite-dimensional Bayesian inverse problems*. JCP 486, 112104, 2023. https://arxiv.org/abs/2210.03008 ; https://doi.org/10.1016/j.jcp.2023.112104 . Access: primary abstract and journal method overview. A residual-based linear variational correction and its accuracy conditions are established prior art; no A18 performance inference is made.
