# 图表的读取与重画

四张图均由 `code/build_report_assets.py` 从同一审计 CSV/JSONL 重画；PNG 用于预览，SVG 用于导出。文稿中的两张核心图由同一结果生成内嵌 pgfplots coordinates，三个 `.tex` 均可独立编译。

- `01_shortlist_coverage`：四对象等权 Cov12。Random 柱为三个固定 seed 的均值，完整 seed 数值在 CSV；误差线表示这三个 seed 的描述性范围，不是置信区间。B2 改善 near-contact/layered，但困难对象缺口仍在；B3 负结果保留。
- `02_query_regret`：损失的正 one-swap gain 比例。E/M/L 分别为 early/middle/late；数字为 k。颜色越暗损失越大。对象的 objective scale 不在此图中混加。
- `03_defect_margin_oracle`：左侧是规定 Fold1 的 near-contact test，右侧是 Fold3 的 smooth validation。后者不是第二个 test。0.60 是机会 gate，0.75 是本次含初始化的 RHS 减半 floor；两者不是实测加速。
- `04_tail_size_and_margin`：左侧每个点是一条合法 Gaussian matched direction，黑线是中位数；大于1的反例保留，A/U不嵌套，tail减小不是逐方向定理。右侧为 B4 的18个 primary finalists，误差/临界半径小于1时通过反事实精度条件。没有正 predicted margin 的项不产生有限半径，仍保留在18的分母中，所以有限横轴处曲线不必达到1。

`04_tail_norms.csv` 与 `04_margin_ratios.csv` 分别提供第四张图的左右数据。重画使用已保存数据，不会运行训练、新 full-wave RHS 或 nonlinear trajectory。

视觉检查：四图已查看。第四张图原来的 log tick 标签重叠已改成0.2、0.5、1、2；最终 PNG/SVG 不再重叠。图形质量检查不代表统计安全校准或 scientific acceptance。
