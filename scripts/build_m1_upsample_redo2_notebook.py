"""Build M1_upsample_redo2.ipynb with inline plots for both requested experiments."""
from textwrap import dedent
import nbformat as nbf
from pathlib import Path

runner_source = Path("scripts/run_m1_upsample_redo2.py").read_text(encoding="utf-8")
runner_source = runner_source.replace("ROOT = Path(__file__).resolve().parents[1]\nif str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))", "ROOT = Path.cwd()")
runner_source = runner_source.split("\nif __name__==\"__main__\":", 1)[0]

def md(s): return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s): return nbf.v4.new_code_cell(dedent(s).strip())

cells=[
 code(runner_source),
 md(r'''# M1 upsample redo 2: fast WERM, 2-fold CV, 100 trials

This notebook reproduces the two requested experiments from `M1_upsample_redo` with the updated core WERM solver. The run uses 36 outer workers and one numerical thread per worker. Every WERM method fits Huber loss and selects the number of change points with **absolute held-out loss** under 2-fold CV.

- $\alpha=0.3$: McScan, unpenalized WERM, exhaustive WERM-L1 joint CV over $C\in\{0.03,0.3,3.0\}$ and 1--4 segments, WERM-L1 at fixed $C=0.3$ with CV over 1--4 segments, DCDP, DPDU, and MOSEG.
- Dense $\alpha=1.0$: McScan and unpenalized Huber WERM only.

The data-generation protocol remains the source notebook's: $p=200$, change points at $(0.35n,0.65n)$, Toeplitz design correlation $\rho=0.4$, and scaled $t_4$ noise. Figures render in notebook output and are not saved separately.'''),
 code('''from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import gaussian_kde
plt.switch_backend("module://matplotlib_inline.backend_inline")
RESULT_FILE = Path(DEFAULT_RESULT_DIR) / "results.pkl"
bundle = run_experiment()
assert set(bundle["experiments"]) == {"alpha03", "alpha10"}
for experiment in bundle["experiments"].values():
    params = experiment["params"]
    assert params["num_folds"] == 2
    assert params["num_trials"] == 100
    assert params["n_jobs"] == 36
    assert params["cv_score_loss"] == "absolute loss"
print(f"Loaded {RESULT_FILE}")'''),
 md('''## Plotting helpers

These reproduce the source notebook's asymmetric-error curves, 10--90% confidence bands, boxplots, and changepoint-location histograms.'''),
 code('''STYLES = {
 "McScan": dict(label="McScan",color="#AA3377",marker="o",linestyle="--",linewidth=1.5),
 "DCDP": dict(label="DCDP",color="#EE6677",marker="s",linestyle="--",linewidth=1.5),
 "DPDU": dict(label="DPDU",color="#4477AA",marker="D",linestyle=":",linewidth=1.5),
 "MOSEG": dict(label="MOSEG",color="#228833",marker="^",linestyle="-.",linewidth=1.5),
 "WERM-unp": dict(label="WERM (Huber; absolute CV)",color="black",marker="*",linestyle="-",linewidth=2.5),
 "WERM-L1-jointCV": dict(label="WERM-L1 joint CV",color="#CCBB44",marker="P",linestyle="-",linewidth=2.0),
 "WERM-L1-C0.3-CV": dict(label="WERM-L1 C=0.3",color="#66CCEE",marker="X",linestyle="--",linewidth=2.0),
}
METRICS=(("haus","Hausdorff distance / n","Location error"),("size","Number of change points","Selected size"),("runtime","Runtime (seconds)","Runtime"))

def _finish_axis(ax,metric,deltas,ylabel,title):
    if metric=="size": ax.axhline(2,color="0.4",linestyle="--",linewidth=1.2); ax.set_ylim(0,4)
    else: ax.set_ylim(bottom=0)
    ax.set(xlabel=r"$\\delta=n/p$",ylabel=ylabel,title=title); ax.set_xticks(deltas)
    ax.spines[["top","right"]].set_visible(False)

def plot_experiment(key):
    experiment=bundle["experiments"][key]; results=experiment["results"]; params=experiment["params"]
    methods=tuple(results["methods"]); deltas=np.asarray(params["delta_list"],float)
    rows=[]
    for i,delta in enumerate(deltas):
        for method in methods:
            rows.append(dict(delta=delta,method=method,mean_hausdorff_over_n=np.nanmean(results[f"haus_{method}"][i]),mean_num_chgpts=np.nanmean(results[f"size_{method}"][i]),mean_runtime_s=np.nanmean(results[f"runtime_{method}"][i]),empty_rate=np.mean(results[f"status_{method}"][i]=="empty_estimate")))
    display(pd.DataFrame(rows))

    fig,axes=plt.subplots(1,3,figsize=(19,5.5))
    for ax,(metric,ylabel,title) in zip(axes,METRICS):
        for method in methods:
            values=np.asarray(results[f"{metric}_{method}"],float); mean=np.nanmean(values,axis=1); q25,q75=np.nanpercentile(values,[25,75],axis=1); style=STYLES[method]
            ax.errorbar(deltas,mean,yerr=np.vstack((np.abs(mean-q25),np.abs(q75-mean))),label=style["label"],color=style["color"],marker=style["marker"],linestyle=style["linestyle"],linewidth=style["linewidth"],capsize=3)
        _finish_axis(ax,metric,deltas,ylabel,title)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=min(4,len(methods)),bbox_to_anchor=(.5,-.08)); fig.suptitle(params["experiment"]); fig.tight_layout(); plt.show()

    fig,axes=plt.subplots(1,3,figsize=(19,5.5))
    for ax,(metric,ylabel,title) in zip(axes,METRICS):
        for method in methods:
            values=np.asarray(results[f"{metric}_{method}"],float); mean=np.nanmean(values,axis=1); q10,q90=np.nanpercentile(values,[10,90],axis=1); style=STYLES[method]
            ax.fill_between(deltas,q10,q90,color=style["color"],alpha=.13); ax.plot(deltas,mean,label=style["label"],color=style["color"],marker=style["marker"],linewidth=style["linewidth"])
        _finish_axis(ax,metric,deltas,ylabel,title+": 10--90% band")
    fig.suptitle(params["experiment"]); fig.tight_layout(); plt.show()

    fig,axes=plt.subplots(1,3,figsize=(19,5.5)); width=.8/max(1,len(methods))
    for ax,(metric,ylabel,title) in zip(axes,METRICS):
        for m,method in enumerate(methods):
            positions=np.arange(len(deltas))+(m-(len(methods)-1)/2)*width
            values=[row[np.isfinite(row)] for row in np.asarray(results[f"{metric}_{method}"],float)]
            artists=ax.boxplot(values,positions=positions,widths=width*.92,patch_artist=True,showfliers=False,medianprops={"color":"white"})
            for box in artists["boxes"]: box.set_facecolor(STYLES[method]["color"])
        ax.set_xticks(np.arange(len(deltas)),[f"{d:g}" for d in deltas]); _finish_axis(ax,metric,np.arange(len(deltas)),ylabel,title+": distributions")
    fig.suptitle(params["experiment"]); fig.tight_layout(); plt.show()

    selected=list(range(len(deltas)))[::4]; fig,axes=plt.subplots(len(selected),len(methods),figsize=(2.8*len(methods),2.6*len(selected)),sharex=True,sharey=True,squeeze=False)
    for row,i in enumerate(selected):
        n=int(deltas[i]*params["p"])
        for col,method in enumerate(methods):
            estimates=[]
            for setting,_,cp in results[f"chgpts_{method}"]:
                if setting==i: estimates.extend(np.asarray(cp,dtype=float)/n)
            ax=axes[row,col]; estimates=np.asarray(estimates)
            if estimates.size:
                ax.hist(estimates,bins=np.linspace(0,1,19),density=True,color=STYLES[method]["color"],alpha=.78,edgecolor="white")
                if estimates.size>6 and np.unique(estimates).size>1:
                    try:
                        grid=np.linspace(0,1,300); ax.plot(grid,gaussian_kde(estimates)(grid),color="black",linewidth=1.8)
                    except np.linalg.LinAlgError: pass
            ax.axvline(.35,color="black",linestyle="--"); ax.axvline(.65,color="black",linestyle="--"); ax.set_xlim(0,1)
            if row==0: ax.set_title(STYLES[method]["label"])
            if col==0: ax.set_ylabel(fr"$\\delta={deltas[i]:g}$"+"\\nDensity")
    fig.supxlabel("Estimated fractional change-point location"); fig.suptitle(params["experiment"]+": changepoint histograms"); fig.tight_layout(); plt.show()'''),
 md(r'''## M1b $\alpha=0.3$'''), code('plot_experiment("alpha03")'),
 md(r'''## M1b $\alpha=1.0$ dense redo'''), code('plot_experiment("alpha10")'),
]
notebook=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"weighted-erm","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}})
nbf.write(notebook,"M1_upsample_redo2.ipynb")
