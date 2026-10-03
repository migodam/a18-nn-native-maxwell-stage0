# Codex 主线程最终科学复核

本记录表示Codex对这一轮Stage 0证据与结论的复核，不是陈教授、用户、审稿人或TAP的acceptance。

- 来源：pinned commit、公开manifest全5556项、原始42模块及36teacher hashes各自通过；所有consumed inputs未改变。REFERENCES恢复与原验证文件缺失明确区分。
- 合同：保存的physical material gauge、六illumination、原始ID、realification、lambda_total与ell均保持；Lambda包含prior和LM damping。Public projected full solve不冒认完整物理truth。
- A：完整字典未截断；B0瓶颈复核；B2的局部改善和B3的负结果并列，随机seeds全报；机会空间支持tiny pilot建议，不支持NN superiority或TAP novelty宣称。
- B：Gaussian合法fullJ与voxel方向支撑逐条核验，saved-step恢复只有一条voxel方向；measurement/current labels不混用。Fold1是规定primary test，Fold3是validation；没有迁移split。
- Oracle：POD/RMS/ridge仅用train对象，source oracle使用评价侧fullJ并明确状态限制。L2 projection不保证全空间最高skip-cardinality；负裁决限定tested pipeline。
- Margin/cost：精确评价侧u与A anchor residual区别明确；10/18 counterfactual skips没有增加anchor机会，且未过60%/75%两个不同gate。初始化、refresh/audit、cache accumulation、模型/source/cold成本和失败均记录。
- 验证：新等价non-Hermitian algebra与真实接口测试通过，未冒称原版复现或未实现NN architecture已合格。独立复现的科学数值完全一致，计时与PDF字节不作相同保证。
- 交付：三份当前TeX原生编译成功，已有Tectonic导出，Poppler全28页实际查看；图表负例保留。后续训练配置disabled，Stage0结束。

仍开放：194条matched voxel vectors、full-current teachers、statistical safety calibration、native architecture实测优势、全流程加速、closest PEC graph全文细节及global novelty。
