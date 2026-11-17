"""Replot saved results only; build a fixed-size publication style specimen.

Run from the repository root with .venv/bin/python tools/build_plot_style_specimen.py.
No experiments, notebook execution, or changes to canonical assets are performed.
"""
from pathlib import Path
import hashlib
import json
import pickle
import subprocess
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

OUT = Path('results/plot_style_specimen_20261102')
OUT.mkdir(parents=True, exist_ok=True)
# Measured from the manuscript's OUP class/options using a standalone TeX probe.
WIDTH_PT = 427.64493
WIDTH = WIDTH_PT / 72.27
HALF = .49 * WIDTH
# Canonical notebook geometry is expressed at 18-pt label size. Scale its
# stroke/marker dimensions by the same factor as the specimen typography.
CANONICAL_SCALE = 10.5 / 18
STYLE = {
    'text.usetex': True, 'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 8.5, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.linewidth': .85, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': .85, 'ytick.major.width': .85,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'savefig.bbox': None, 'figure.dpi': 120, 'savefig.dpi': 200,
}
plt.rcParams.update(STYLE)
sources = {}
audit = []

def load(path):
    p = Path(path)
    sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open('rb') as f:
        return pickle.load(f)

def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Verify all axis labels/ticks/titles fit on the fixed canvas.
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (name, box)
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (name, box)
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.png'))
    audit.append({'name': name, 'inches': fig.get_size_inches().tolist(),
                  'axes': [list(a.get_position().bounds) for a in fig.axes]})
    plt.close(fig)

def half_panel():
    fig, ax = plt.subplots(figsize=(HALF, 2.05))
    fig.subplots_adjust(left=.19, right=.98, bottom=.23, top=.97)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    return fig, ax

m2 = load('results/M2_upsample_final_alpha03_seed42_100trials.pkl')
r, p = m2['results'], m2['params']
fig, ax = half_panel()
for key, label, color, marker, ls in [
    ('McScan', 'McScan', '#AA3377', 'o', '--'),
    ('charcoal', 'charcoal', '#336677', 'v', '-.'),
    ('WERM-unp-10fold', r'\textbf{WERM}', 'black', 'D', '-')]:
    idx = np.arange(len(p['delta_list']))[1:-2:1 if key == 'charcoal' else 2]
    if key == 'charcoal': idx = idx[np.asarray(p['delta_list'])[idx] < 9.5]
    values = np.asarray(r['size_' + key])[idx]
    mean = np.nanmean(values, axis=1)
    q25, q75 = np.nanpercentile(values, [25, 75], axis=1)
    ax.errorbar(np.asarray(p['delta_list'])[idx], mean,
                yerr=[abs(mean-q25), abs(q75-mean)], label=label,
                color=color, marker=marker, ls=ls,
                linewidth=3.5*CANONICAL_SCALE,
                markersize=8*CANONICAL_SCALE, capsize=4*CANONICAL_SCALE,
                capthick=2*CANONICAL_SCALE,
                markeredgecolor='white', markeredgewidth=2*CANONICAL_SCALE,
                alpha=.8)
ax.axhline(len(p['fractional_chgpt_locations']), color='gray',
           lw=2.5*CANONICAL_SCALE, alpha=.8, zorder=0)
ax.set(xlabel=r'$\delta=n/p$', ylabel='No. change points', ylim=(0, p['L']+1))
ax.set_yticks([0, 2, 4])
ax.set_xticks(np.asarray(p['delta_list'])[1:-2:2][::2])
ax.legend(loc='upper right', frameon=False, handlelength=1.8, labelspacing=.2)
save(fig, 'm2_counts')

m3 = load('results/M3_usample_final_delta3_alpha01_GLMBSA_WERM-L1_fullCV2_seed42_100trials.pkl')
r, p = m3['results'], m3['params']
idx = np.flatnonzero(np.asarray(p['kappa_list']) >= 1.5)
fig, ax = half_panel()
rng = np.random.default_rng(42)
handles = []
for j, (key, label, color) in enumerate([
    ('GLMBSA-CV', 'GLMBSA', '#E69F00'),
    ('WERM-L1-fullCV-2fold', r'\textbf{WERM+L1}', 'black')]):
    handles.append(Patch(facecolor=color, alpha=.28, edgecolor='black', label=label))
    for k, source_index in enumerate(idx):
        values = np.asarray(r['haus_' + key][source_index], dtype=float)
        values = values[(np.asarray(r['size_' + key][source_index]) > 0) & np.isfinite(values)]
        assert len(values)
        pos = k*3+j
        bp = ax.boxplot([values], positions=[pos], widths=.65, patch_artist=True,
                        showfliers=False, whis=1.5, autorange=True, manage_ticks=False,
                        medianprops={'color':'black', 'linewidth':1.25},
                        boxprops={'linewidth':.75}, whiskerprops={'linewidth':.7},
                        capprops={'linewidth':.7})
        bp['boxes'][0].set(facecolor=color, alpha=.28)
        ax.scatter(pos+rng.uniform(-.19,.19,len(values)), values,
                   s=3, alpha=.12, color=color, linewidths=0, zorder=3)
ax.set_xticks(np.arange(len(idx))[::2]*3+.5,
              [f'{v:.0f}' for v in np.square(np.asarray(p['kappa_list'])[idx])[::2]])
ax.set(xlabel=r'$\kappa^2$', ylabel=r'Hausdorff dist. / $n$')
ax.legend(handles=handles, frameon=False, loc='upper right', handlelength=1.1,
          labelspacing=.2)
save(fig, 'm3_location')

prior = load('results/E_PRIOR_jointconfigs_actualgap_100paired_seed20260916.pkl')
names = ['Bad', 'Neutral', 'Good']
colors = ['#EE6677', '#4477AA', '#228833']
n = prior['params']['N']
truth = np.asarray(prior['params']['true_cp'])/n

def triptych():
    fig, axes = plt.subplots(1,3,figsize=(WIDTH,1.5),sharey=True)
    fig.subplots_adjust(left=.11, right=.985, bottom=.31, top=.80, wspace=.20)
    for ax, name in zip(axes,names):
        ax.set_title(name, pad=3)
        ax.set(xlim=(-.02,1.02), xlabel='Fractional location')
        ax.set_xticks([0,.5,1], ['0','0.5','1'])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        for cp in truth: ax.axvline(cp,color='black',ls='--',lw=.75,alpha=.65)
    return fig, axes

fig, axes = triptych()
for ax,name,color in zip(axes,names,colors):
    configs = np.asarray(prior['configs'][name],dtype=int)
    mass = np.bincount(configs.ravel(),minlength=n)/(2*len(configs))
    assert np.isclose(mass.sum(),1)
    ax.plot(np.arange(1,n)/n,mass[1:],color=color,lw=1.3)
axes[0].set_ylabel('Prior prob.')
save(fig,'prior_triptych')

# Canonical M1 tighter-layout histogram syntax: 12 data-range bins, five-fold
# CV Gaussian KDE, grey KDE/fill, fixed density scale, truth-position ticks,
# tightly shared axes, and figure-level labels. Prior cases retain their
# meaningful Bad/Neutral/Good order; there is no delta row label here.
fig, axes = plt.subplots(1,3,figsize=(WIDTH,1.5),sharex=True,sharey=True,
                         gridspec_kw={'wspace':.05,'hspace':.08})
fig.subplots_adjust(left=.11,right=.985,bottom=.27,top=.80)
for ax,name,color in zip(axes,names,colors):
    values = np.concatenate([np.asarray(row[name]['cp'])/n for row in prior['trials']])
    ax.hist(values,bins=12,alpha=.8,color=color,density=True,
            edgecolor='white',linewidth=1.0*CANONICAL_SCALE)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth':np.logspace(-2,0,50)},cv=5)
    grid.fit(values.reshape(-1,1))
    x_range = np.linspace(0,1,200)
    density = np.exp(grid.best_estimator_.score_samples(x_range.reshape(-1,1)))
    ax.plot(x_range,density,color='#555555',linewidth=1.0*CANONICAL_SCALE,
            linestyle='-',alpha=.8,zorder=2)
    ax.fill_between(x_range,density,color='#999999',alpha=.2,zorder=1)
    for cp in truth:
        ax.axvline(cp,color='black',linestyle='--',linewidth=2*CANONICAL_SCALE,
                   alpha=.9,zorder=10)
    ax.set(xlim=(0,1),ylim=(0,18))
    ax.set_xticks(truth,[f'${cp:.2f}$' for cp in truth])
    ax.set_yticks([0,8,16])
    ax.tick_params(axis='both',which='major',width=CANONICAL_SCALE,
                   length=4*CANONICAL_SCALE,labelsize=14*CANONICAL_SCALE)
    ax.spines['left'].set_linewidth(CANONICAL_SCALE)
    ax.spines['bottom'].set_linewidth(CANONICAL_SCALE)
    ax.set_title(name,fontsize=18*CANONICAL_SCALE,pad=10*CANONICAL_SCALE)
fig.text(.5,.01,'Fractional location',ha='center',va='bottom',fontsize=10.5)
fig.text(.015,.5,'Density',va='center',rotation='vertical',fontsize=10.5)
save(fig,'location_histograms')

mi = load('results/myocardial_final_redo_e3d965cc4b9b/scenario_100_100_q10_seed123.pkl')
ages, probs = np.asarray(mi['ages']), np.asarray(mi['conditional'])
finite = np.isfinite(ages) & np.isfinite(probs)
assert np.isclose(probs[finite].sum(),1)
fig, ax = plt.subplots(figsize=(WIDTH,1.18))
fig.subplots_adjust(left=.11,right=.985,bottom=.39,top=.96)
markers, stems, _ = ax.stem(ages[finite],probs[finite],linefmt='k-',markerfmt='ko',basefmt=' ')
plt.setp(stems,linewidth=1.05,alpha=.8)
plt.setp(markers,markersize=2.3,markeredgewidth=0)
ax.set(xlim=(26,95.1), xlabel='Age (years)')
ax.set_xticks([26,40,50,60,75,81,92])
ax.set_yticks([0,max(probs[finite])],['0',f'{max(probs[finite]):.2f}'])
ax.grid(axis='y',ls=':',color='gray',alpha=.4)
save(fig,'myocardial_posterior')

tex = r'''\documentclass[12pt]{article}
\usepackage[paperwidth=188.88mm,paperheight=246mm,textwidth=427.64493pt,textheight=596.5pt,centering]{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[T1]{fontenc}
\captionsetup{font=small,skip=2pt}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
{\small\textbf{Publication style specimen}\hfill Saved results only}\par
\vspace{3pt}
\begin{subfigure}[t]{.49\linewidth}
\includegraphics[width=\linewidth]{m2_counts.pdf}
\caption*{(a) M2: selected counts.}
\end{subfigure}\hfill
\begin{subfigure}[t]{.49\linewidth}
\includegraphics[width=\linewidth]{m3_location.pdf}
\caption*{(b) M3: localization.}
\end{subfigure}
\par\vspace{4pt}
\includegraphics[width=\linewidth]{prior_triptych.pdf}
\par{\small (c) E-PRIOR: exact pooled change-point location prior.}\par
\vspace{3pt}
\includegraphics[width=\linewidth]{location_histograms.pdf}
\par{\small (d) E-PRIOR: estimated locations; dashed lines mark truth.}\par
\vspace{3pt}
\includegraphics[width=\linewidth]{myocardial_posterior.pdf}
\par{\small (e) Myocardial: posterior conditional on one change point.}\par
\vspace{5pt}
{\footnotesize Fixed canvases at manuscript text width; Computer Modern;
axis labels 10.5\,pt, ticks 8.5\,pt. View at actual size to assess typography.
}
\end{document}
'''
# subfigure captions need a figure context without introducing a floating page.
tex = tex.replace(r'\usepackage{subcaption}',r'\usepackage{subcaption}')
tex = tex.replace(r'\vspace{3pt}'+'\n'+r'\begin{subfigure}',
                  r'\vspace{3pt}'+'\n'+r'\begin{minipage}{\linewidth}\captionsetup{type=figure}'+'\n'+r'\begin{subfigure}',1)
tex = tex.replace(r'\par\vspace{4pt}', r'\end{minipage}\par\vspace{4pt}',1)
(OUT/'specimen.tex').write_text(tex)
(OUT/'manifest.json').write_text(json.dumps({'source_sha256':sources,'style':STYLE,
    'text_width_tex_pt':WIDTH_PT,'figures':audit},indent=2))
subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','specimen.tex'],
               cwd=OUT,check=True,stdout=(OUT/'compile.stdout').open('w'))
subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile','specimen.pdf','specimen'],
               cwd=OUT,check=True)
print('Created:',OUT/'specimen.pdf')
