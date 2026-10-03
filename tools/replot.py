"""Replot the four research figures from archived CSVs into a new directory."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
assert not out.exists(), "Use a new output directory."
out.mkdir(parents=True)
def rows(name):
    with (ROOT / name).open(newline="") as f:
        return list(csv.DictReader(f))
def save(fig, name):
    fig.savefig(out / (name+".png"), bbox_inches="tight", dpi=220)
    fig.savefig(out / (name+".svg"), bbox_inches="tight")
    plt.close(fig)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
cfg = json.loads((ROOT / "stage0_config.json").read_text())
oids = cfg["object_order"]; names = [cfg["object_names"][str(i)] for i in oids]
A = rows("results/A18_BASELINE_OBJECT_SUMMARY.csv")
fig, ax = plt.subplots(figsize=(8.4, 4.1)); x = np.arange(4); width = .18
colors = ["#6a7b8c", "#a494b8", "#267e9a", "#cd715c"]
for pos, method in enumerate(["B0", "B1", "B2", "B3"]):
    data = np.array([[float(r["Cov12"]) for r in A if r["method"] == method and r["object_id"] == str(oid)] for oid in oids])
    values = data.mean(axis=1)
    yerr = [values-data.min(axis=1), data.max(axis=1)-values] if method == "B1" else None
    ax.bar(x+(pos-1.5)*width, values, width, yerr=yerr, capsize=3, color=colors[pos],
           label=method+(" random: 3-seed mean/range" if method == "B1" else ""))
ax.axhline(1, color="#404040", ls=":", label="B5 one-swap teacher")
ax.scatter([1, 3], [.7, .85], marker="_", s=700, c="#101010", label="Object-specific future gates")
ax.set_xticks(x, names); ax.set_ylim(0, 1.07); ax.set_ylabel("Gain-weighted Cov12")
ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.27)); fig.subplots_adjust(top=.77)
save(fig, "01_shortlist_coverage")
R = rows("figures/02_query_regret.csv")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
for ax, method in zip(axes, ["B0", "B2", "B3"]):
    values = [[float(r["normalized_gain_regret"]) for r in R if r["method"] == method and r["object_id"] == str(oid)] for oid in oids]
    im = ax.imshow(values, vmin=0, vmax=1, cmap="magma_r", aspect="auto")
    ax.set_title(method); ax.set_xticks(range(9), ["E4", "E8", "E16", "M4", "M8", "M16", "L4", "L8", "L16"], rotation=45)
    ax.set_yticks(range(4), names)
fig.colorbar(im, ax=axes, label="Lost fraction of positive one-swap gain", shrink=.8, pad=.02)
save(fig, "02_query_regret")
R = rows("figures/03_defect_margin_oracle.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, part in zip(axes, ["test", "validation"]):
    for target, color in [("direct_defect", "#cd715c"), ("anchor_tail", "#267e9a")]:
        for model, ls in [("POD_EXACT_LATENT_ORACLE", "-"), ("FIXED_LINEAR_RIDGE", "--")]:
            rr = sorted([r for r in R if r["partition"] == part and r["target"] == target and r["model"] == model], key=lambda r:int(r["rank"]))
            ax.plot([int(r["rank"]) for r in rr], [float(r["skip_fraction"]) for r in rr], marker="o", ls=ls, c=color,
                    label=target.replace("_", " ")+" / "+("POD oracle" if model.startswith("POD") else "ridge"))
    ax.axhline(.6, c="#888888", ls=":", label="Stage 0 opportunity gate")
    ax.axhline(.75, c="#222222", ls="-.", label="Matched RHS-halving floor")
    ax.set_xticks([8,16,32]); ax.set_ylim(-.03,1.02); ax.set_xlabel("POD output rank")
    ax.set_title("near-contact (test, fold 1)" if part=="test" else "smooth (validation, fold 3)")
axes[0].set_ylabel("Counterfactual accept-check skips"); axes[1].legend(fontsize=6.8, loc="upper right")
save(fig, "03_defect_margin_oracle")
N = rows("figures/04_tail_norms.csv"); M = rows("figures/04_margin_ratios.csv")
fig, axes = plt.subplots(1,2,figsize=(8.7,3.5))
for pos, oid in enumerate([2001,2007]):
    values=[float(r["tail_over_direct"]) for r in N if r["object_id"]==str(oid)]
    axes[0].scatter(np.full(len(values),pos)+np.linspace(-.12,.12,len(values)),values,s=8,alpha=.35,c="#267e9a")
    axes[0].scatter([pos],[np.median(values)],marker="_",s=1000,c="#121212")
axes[0].set_xticks([0,1],["smooth","near-contact"]); axes[0].set_ylabel("||anchor tail|| / ||direct defect||")
axes[0].set_title("Known correction reduces target size")
for rank in [8,16,32]:
    rr=[r for r in M if r["rank"]==str(rank)]; assert len(rr)==18
    values=sorted(float(r["margin_ratio"]) for r in rr if r["margin_ratio"]!="")
    axes[1].step(values,np.arange(1,len(values)+1)/len(rr),where="post",label=f"rank {rank}")
axes[1].axvline(1,c="#222222",ls=":"); axes[1].set_xscale("log")
axes[1].set_xticks([.2,.5,1,2],["0.2","0.5","1","2"]); axes[1].minorticks_off(); axes[1].set_ylim(0,1)
axes[1].set_xlabel("Response error / epsilon_crit"); axes[1].set_ylabel("Fraction of all 18 finalists")
axes[1].set_title("near-contact test, B4 finalists"); axes[1].legend(fontsize=8)
save(fig,"04_tail_size_and_margin")
print(json.dumps({"figures":4,"formats":["png","svg"],"new_full_wave_rhs":0,"training_epochs":0,"scope":"CSV REPLOT; NO PIXEL IDENTITY CLAIM"}))
