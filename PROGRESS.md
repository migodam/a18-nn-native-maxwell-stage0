# Stage 0 完成与恢复入口

1. 重读 inputs/A18_CODEX_PROMPT.md、START_HERE.md 和 STATUS.json。
2. 本轮已完成并停止：九项审计、A/B独立裁决、三份TeX/PDF、28页视觉检查、全来源哈希和独立数值复现。
3. A_GO_TINY_PILOT仅为下一阶段建议；B_DATA_INCOMPLETE与Gaussian所测表示的negative结果分列。不得因恢复/compact自动开始训练或补标签。
4. 新full-wave RHS、NN epochs、new nonlinear trajectories、GPU training均为零。后续FUTURE_TRAINING_CONFIG enabled=false。
5. 主线程为scientific owner，provider/model未变；所有失败与费用保留。改变科学输入或文稿后须重新绑定相应验证与最终哈希，不能沿用旧QA。
