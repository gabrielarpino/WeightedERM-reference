"""Append and execute only new saved-result plotting cells in final notebooks."""
from pathlib import Path
import copy
import hashlib
import json
import re
import sys
import tempfile
import shutil
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/consistent_plots_20261102'
TAG='updated-consistent-plots-20261102'
HELPERS=(ROOT/'tools/consistent_plotting.py').read_text()

def section(title, code):
    return title, code.strip()

PLAN={}
PLAN['M2_upsample_final.ipynb']=('M2',[
 section('Original three-method experiment — Canonical M2 Comparison Experiment plots',r'''
# This is the artifact loaded by the original canonical plotting cell, NOT the
# fresh WERM+P experiment. Each comparison keeps its own saved fits and timings.
original=load_saved('results/M2_upsample_final_alpha03_seed42_100trials.pkl')
results,params=original['results'],original['params']
deltas=np.asarray(params['delta_list']);truth=np.asarray(params['fractional_chgpt_locations'])
methods=['McScan','charcoal','WERM-unp-10fold']
curve_indices=np.arange(len(deltas))[1:-2:2]
charcoal_indices=np.arange(len(deltas))[1:-2]
charcoal_indices=charcoal_indices[deltas[charcoal_indices]<9.5]
histogram_indices=np.arange(len(deltas))[4:9:2]
assert params['num_trials']==100
np.testing.assert_allclose(deltas[histogram_indices],[4.5,6.5,8.5])
print('Original M2: McScan, charcoal, WERM (unpenalized, 10-fold CV); 100 trials per setting.')
for metric,ylabel in [('haus',r'Hausdorff dist. / $n$'),('size','No. change points'),('runtime','Runtime (s)')]:
    for kind in ('line','box'):
        with plt.rc_context(CANONICAL_STYLE if kind=='line' else PAPER_STYLE):
            if kind=='line':
                fig,ax=plt.subplots(figsize=(5.,4.));fig._canonical_m1=True
            else:fig,ax=paper_panel()
            rng=np.random.default_rng(42);handles=[]
            for j,method in enumerate(methods):
                label,color,marker,ls=METHODS[method]
                indices=charcoal_indices if method=='charcoal' else curve_indices
                values=np.asarray(results[f'{metric}_{method}'],float)[indices].copy()
                assert values.shape[1]==100
                if metric=='haus':
                    sizes=np.asarray(results[f'size_{method}'])[indices]
                    values=np.where(sizes>0,values,np.nan)
                if kind=='line':
                    error_series(ax,deltas[indices],values,label,color,marker,ls,scale=1)
                else:
                    for i,v in zip(indices,values):
                        box_group(ax,v,float(deltas[i])+(j-1)*.20,color,rng,width=.18,jitter=.06)
                    handles.append(Patch(facecolor=color,alpha=.28,edgecolor='black',label=label))
            ax.set(xlabel=r'$\delta=n/p$',ylabel=ylabel)
            ax.set_xticks(deltas[curve_indices][::2])
            if metric=='size':
                ax.axhline(len(truth),color='gray',lw=2.5*(1 if kind=='line' else SCALE),alpha=.8,zorder=0)
                ax.set_ylim(0,params['L']+1);ax.yaxis.set_major_locator(MaxNLocator(4,integer=True))
            if metric=='runtime':
                ax.set_ylim(bottom=0)
                ax.legend(handles=handles if kind=='box' else None,loc='best',
                          prop={'size':14 if kind=='line' else 8},fancybox=True,frameon=True,framealpha=.4)
            if kind=='line':fig.tight_layout()
            export_plot(fig,f'M2_original_{metric}_{kind}')
with plt.rc_context(PAPER_STYLE):
    keys=list(reversed(methods));groups=[]
    for i in histogram_indices:
        n=int(deltas[i]*params['p']);row=[]
        for method in keys:
            parts=[np.asarray(cp).ravel() for index,cp in results[f'chgpts_{method}'] if index==i]
            cps=np.concatenate(parts) if parts else np.array([])
            row.append(cps[cps!=-1]/n)
        groups.append(row)
    histogram_grid(groups,[METHODS[k][0] for k in keys],[METHODS[k][1] for k in keys],truth,
        'M2_original_hist',row_labels=[rf'$\delta={deltas[i]:g}$' for i in histogram_indices])
'''),
 section('Completed fresh four-method comparison, gamma = 6',r'''
saved=load_saved('results/M2_upsample_final_WERM_plus_P_gamma6_fresh/comparison.pkl')
params=saved['params'];deltas=np.asarray(params['delta_list']);truth=np.asarray(params['fractional_chgpt_locations'])
methods=['McScan','charcoal','WERM','WERM+P']
METHODS['WERM']=(r'\textbf{WERM+CV}','black','D','-')
METHODS['WERM+P']=(r'\textbf{WERM+P}','#228833','h','-')
records=saved['trial_records']
failed={(r['setting'],r['trial']) for r in records if 'charcoal_R_error' in r.get('diagnostics',{})}
assert len(failed)==saved['charcoal_R_errors']==4
print('Excluding charcoal R failures from plotted summaries:',sorted(failed))
assert len({(r['setting'],r['trial']) for r in records})==len(records)
print('Method | attempted trials | successful | empty | R/errors')
for method in methods:
    indices=saved['charcoal_indices'] if method=='charcoal' else saved['setting_indices']
    entries=[r['estimates'][method] for r in records if r['setting'] in indices and method in r['estimates']]
    assert len(entries)==100*len(indices)
    for i in indices:
        assert {r['trial'] for r in records if r['setting']==i and method in r['estimates']}==set(range(100))
    errors=[e for e in entries if e['raw_cp'] is None]
    r_failures=len(failed) if method=='charcoal' else 0
    print(method,len(entries),len(entries)-len(errors)-r_failures,
          sum(e['size']==0 for e in entries)-r_failures,len(errors)+r_failures)
    for e in entries:
        if e['raw_cp'] is None:assert not np.isfinite(e['haus'])
        elif e['size']==0:assert not np.isfinite(e['haus'])
print('The saved artifact encoded four R errors as empty estimates. The new plots exclude them; valid empty estimates remain zero counts. Saved data are unchanged.')
with plt.rc_context(PAPER_STYLE):
    for metric,ylabel in [('haus',r'Hausdorff dist. / $n$'),('size','No. change points'),('runtime','Runtime (s)')]:
        for kind in ('line','box'):
            fig,ax=paper_panel();rng=np.random.default_rng(42);handles=[]
            for j,method in enumerate(methods):
                label,color,marker,ls=METHODS[method]
                indices=saved['charcoal_indices'] if method=='charcoal' else saved['curve_indices']
                source=np.asarray(saved['metrics'][method][metric]).copy()
                if method=='charcoal':
                    for setting,trial in failed:source[setting,trial]=np.nan
                values=source[indices]
                if kind=='line':error_series(ax,deltas[indices],values,label,color,marker,ls)
                else:
                    for i,v in zip(indices,values):box_group(ax,v,float(deltas[i])+(j-1.5)*.17,color,rng,width=.14,jitter=.055)
                    handles.append(Patch(facecolor=color,alpha=.28,edgecolor='black',label=label))
            ax.set(xlabel=r'$\delta=n/p$',ylabel=ylabel)
            ax.set_xticks(deltas[saved['curve_indices']][::2])
            if metric=='size':
                ax.axhline(2,color='gray',lw=2.5*SCALE,alpha=.8,zorder=0)
                ax.yaxis.set_major_locator(MaxNLocator(4,integer=True));ax.set_ylim(0,5)
            if metric=='runtime':
                ax.legend(handles=handles if kind=='box' else None,loc='best',
                          prop={'size':8},fancybox=True,frameon=True,framealpha=.4)
            export_plot(fig,f'M2_fresh_{metric}_{kind}')
    for page,keys in enumerate([['WERM+P','WERM','charcoal','McScan']],1):
        groups=[]
        for i in saved['histogram_indices']:
            n=int(deltas[i]*params['p']);group=[]
            for method in keys:
                cps=[np.asarray(r['estimates'][method]['raw_cp'])/n for r in records
                     if r['setting']==i and method in r['estimates'] and r['estimates'][method]['raw_cp'] is not None]
                group.append(np.concatenate(cps))
            groups.append(group)
        histogram_grid(groups,[METHODS[k][0] for k in keys],[METHODS[k][1] for k in keys],truth,
            'M2_fresh_hist',row_labels=[rf'$\delta={deltas[i]:g}$' for i in saved['histogram_indices']])
''')])
PLAN['M1_upsample_final.ipynb']=('M1',[
 section('M1b: alpha = 0.3 — canonical comparison',r'''
with plt.rc_context(PAPER_STYLE):
    comparison_plots('results/M1_upsample_final_alpha03_seed46_100trials.pkl',
        'M1b_alpha03', ['McScan','DCDP','DPDU','MOSEG','WERM-L1-jointCV'], min_x=2)
'''),
 section('M1b: alpha = 0.3 — WERM+L1, DCDP and DPDU',r'''
with plt.rc_context(PAPER_STYLE):
    comparison_plots('results/M1_upsample_final_alpha03_seed46_100trials.pkl',
        'M1b_alpha03_top3', ['DCDP','DPDU','WERM-L1-jointCV'], min_x=2,
        metrics=('haus','size','size_error','runtime'))
'''),
 section('M1b dense: alpha = 1.0 — Student-t noise',r'''
with plt.rc_context(PAPER_STYLE):
    comparison_plots('results/M1_upsample_final_alpha10_dense_seed46_100trials.pkl',
        'M1b_dense_student', ['McScan','WERM-unp-10fold'], min_x=2)
'''),
 section('M1a: fixed-count comparison',r'''
with plt.rc_context(PAPER_STYLE):
    comparison_plots('results/M1_upsample_final_M1a_native_fixedcount_seed42_100trials.pkl',
        'M1a_fixed_count', ['McScan','MOSEG','WERM-unp'], min_x=2)
'''),
 section('M1b dense: alpha = 1.0 — Gaussian noise, least-squares WERM',r'''
with plt.rc_context(PAPER_STYLE):
    comparison_plots('results/M1_upsample_final_alpha10_dense_gaussian_ls_seed46_100trials.pkl',
        'M1b_dense_gaussian_ls', ['McScan','WERM-unp-10fold'], min_x=2)
''')])
PLAN['M3_upsample_final.ipynb']=('M3',[
 section(f'M3: delta = {delta}, alpha = 0.1',f'''
with plt.rc_context(PAPER_STYLE):
    comparison_plots('results/M3_usample_final_delta{delta}_alpha01_GLMBSA_WERM-L1_fullCV2_seed42_100trials.pkl',
        'M3_delta{delta}', ['GLMBSA-CV','WERM-L1-fullCV-2fold'],
        min_x=1.5, m3=True, histogram_stride=1)
''') for delta in (3,6)])

PLAN['E_L_misspecified_L.ipynb']=('E_L_fixed',[
 section('Fixed working number of signals',r'''
saved=load_saved('results/E_L_fixedL_L1to6_100paired_seed20260916.pkl')
trials,params=saved['trials'],saved['params']
Ls=np.asarray(params['working_L']);n=params['N'];truth=np.asarray(params['true_cp'])/n
groups=[[r[int(L)]['conditional_haus'] for r in trials] for L in Ls]
print('Working L-hat | finite trials | mean | median')
for L,v in zip(Ls,groups):
    v=clean(v);print(L,len(v),np.mean(v) if len(v) else 'undefined',np.median(v) if len(v) else 'undefined')
with plt.rc_context(PAPER_STYLE):
    simple_summary(groups,Ls,r'Working $\hat{L}$',r'Hausdorff dist. / $n$',
        'fixed_L_location_line',kind='line',error='se',title='Location error')
    simple_summary(groups,Ls,r'Working $\hat{L}$',r'Hausdorff dist. / $n$',
        'fixed_L_location_box',colors=['#4477AA' if L<4 else 'black' if L==4 else '#EE6677' for L in Ls],
        title='Location error')
    pools=[np.concatenate([np.asarray(r[int(L)]['cp'])/n for r in trials]) for L in Ls]
    histogram_grid([pools],['']*len(Ls),['black']*len(Ls),truth,'fixed_L_histograms',
        panel_titles=[[rf'Working $\hat{{L}}={L}$' for L in Ls]])
''')])

PLAN['E_L_upper_bound.ipynb']=('E_L_upper_bound',[
 section('Upper-bound sensitivity: n = 1000, p = 100',r'''
study=load_saved('results/E_L_upper_bound_n1000_p100_gap10/study.pkl')
assert study['settings']==[(1000,100)]
rows=study['rows'][(1000,100)];assert len(rows)==100
Ls=np.arange(1,7);truth=np.array([.2,.5,.8])
with plt.rc_context(PAPER_STYLE):
    for metric,ylabel in [('haus',r'Hausdorff dist. / $n$'),('count','No. change points')]:
        groups=[[r['fits'][int(L)]['selected']-1 if metric=='count' else r['fits'][int(L)]['haus'] for r in rows] for L in Ls]
        for kind in ('line','box'):
            simple_summary(groups,Ls,r'$L$',ylabel,f'upper_L_{metric}_{kind}',kind=kind,
                           truth=3 if metric=='count' else None)
    scores=np.array([r['fits'][6]['scores'] for r in rows])/1000
    simple_summary(scores.T,Ls,r'$L$','CV absolute error','upper_L_cv_scores',kind='line')
    pools=[np.concatenate([np.asarray(r['fits'][int(L)]['cp'])/1000 for r in rows]) for L in Ls]
    histogram_grid([pools],['']*len(Ls),['black']*len(Ls),truth,'upper_L_histograms',
                   panel_titles=[[rf'$L={L}$' for L in Ls]])
''')])

PLAN['E_PRIOR_misspecified_prior.ipynb']=('E_PRIOR',[
 section('Bad, neutral and good priors',r'''
saved=load_saved('results/E_PRIOR_jointconfigs_actualgap_100paired_seed20260916.pkl')
trials,params=saved['trials'],saved['params'];n=params['N']
names=['Bad','Neutral','Good'];colors=['#EE6677','#4477AA','#228833']
truth=np.asarray(params['true_cp'])/n
groups=[[row[name]['conditional_haus'] for row in trials] for name in names]
print('Prior | mean Hausdorff / n | median Hausdorff / n')
for name,values in zip(names,groups):print(name,np.nanmean(values),np.nanmedian(values))
with plt.rc_context(PAPER_STYLE):
    simple_summary(groups,np.arange(3),'',r'Hausdorff dist. / $n$',
        'prior_location_line',kind='line',error='se',labels=names)
    simple_summary(groups,np.arange(3),'',r'Hausdorff dist. / $n$',
        'prior_location_box',colors=colors,labels=names)

    # Exact pooled prior masses and exact signal marginals: no resimulation or smoothing.
    for plot in ('pooled_prior','signal_marginals'):
        fig,axes=plt.subplots(1,3,figsize=(PAPER_WIDTH,1.85 if plot=='signal_marginals' else 1.5),sharey=True)
        fig.subplots_adjust(left=.11,right=.985,bottom=.27 if plot=='signal_marginals' else .31,
                            top=.84 if plot=='signal_marginals' else .80,wspace=.20)
        for ax,name,color in zip(axes,names,colors):
            if plot=='pooled_prior':
                configs=np.asarray(saved['configs'][name],int)
                mass=np.bincount(configs.ravel(),minlength=n)/(2*len(configs))
                assert np.isclose(mass.sum(),1)
                ax.plot(np.arange(1,n)/n,mass[1:],color=color,lw=1.3)
            else:
                marginal=np.asarray(saved['marginals'][name])
                np.testing.assert_allclose(marginal.sum(axis=0),1,atol=1e-8)
                for l,ls in enumerate(('-','--',':')):
                    ax.plot(np.arange(n)/n,marginal[l],color=color,ls=ls,lw=1.3,
                            label=rf'$\pi_i^{{({l+1})}}$')
            for cp in truth:ax.axvline(cp,color='black',ls='--',lw=.75,alpha=.65)
            ax.set(xlim=(-.02,1.02),xlabel='Fractional location')
            ax.set_title(name,pad=3);ax.set_xticks([0,.5,1],['0','0.5','1'])
            ax.yaxis.set_major_locator(MaxNLocator(3))
        axes[0].set_ylabel('Prior prob.' if plot=='pooled_prior' else 'Sample weight')
        if plot=='signal_marginals':axes[-1].legend(fontsize=7,loc='center right')
        export_plot(fig,plot)

    # Keep the exact joint-prior visualization; each pixel is a configuration.
    fig,axes=plt.subplots(1,3,figsize=(PAPER_WIDTH,2.6))
    for col,(ax,name) in enumerate(zip(axes,names)):
        left=.11+col*.295
        ax.set_position([left,.32,.25,.25*PAPER_WIDTH/2.6])
        configs=np.asarray(saved['configs'][name],int);mass=np.zeros((n,n))
        np.add.at(mass,(configs[:,0],configs[:,1]),1/len(configs))
        im=ax.imshow(mass.T,origin='lower',interpolation='none',extent=(0,1,0,1),cmap='magma',aspect='equal')
        ax.scatter(*truth,marker='x',color='#22D3EE',s=18,linewidth=1)
        ax.set_title(name,pad=4);ax.set_xlabel('Fractional location')
        ax.set_xticks([0,.5,1],['0','0.5','1']);ax.set_yticks([0,.5,1],['0','0.5','1'])
        cax=fig.add_axes([left,.10,.25,.035])
        cbar=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,mass.max()],format='%.0e')
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.get_xticklabels()[0].set_ha('left')
        cbar.ax.get_xticklabels()[-1].set_ha('right')
    axes[0].set_ylabel('Fractional location')
    export_plot(fig,'joint_configuration_priors')

    pools=[np.concatenate([np.asarray(row[name]['cp'])/n for row in trials]) for name in names]
    histogram_grid([pools],names,colors,truth,'prior_estimated_locations',density_label_x=.015)
    # Ordered estimates retain both coordinates, now in the same canonical grid
    # format instead of the older overlaid step-histogram format.
    ordered=[[np.array([row[name]['cp'][coordinate]/n for row in trials])
              for name in names] for coordinate in (0,1)]
    histogram_grid(ordered,names,colors,truth,'prior_ordered_estimates',
                   row_labels=['First change','Second change'])
''')])

PLAN['E_search_penalty.ipynb']=('E_search_penalty_M3',[
 section('Genuine M3 logistic comparison — independent assessment only',r'''
from scipy.special import gammaln
saved=load_saved('results/M3_genuine_logbinomial/rerun_assessment100.pkl')
rows=saved['assessment'];design=saved['design'];n=design['n'];truth=np.asarray(design['true_cp'])
assert len(rows)==100
assert not ({r['data_hash'] for r in saved['calibration']}&{r['data_hash'] for r in rows})
Ks=np.arange(1,5);logcomb=gammaln(n)-gammaln(Ks)-gammaln(n-Ks+1)
cps=[[],[]]
for row in rows:
    k=int(np.argmin(row['raw_by_K']+saved['gamma']*logcomb)+1)
    assert k==row['penalties']['log_binomial']['K']
    cps[0].append(np.asarray(row['cv_cp']))
    cps[1].append(np.asarray(row['cp_by_K'][k]))
labels=[r'\textbf{WERM+CV}',r'\textbf{WERM+P}'];colors=['black','#4477AA']
counts=[np.array([len(cp) for cp in method]) for method in cps]
H=[]
for method in cps:
    values=[]
    for cp in method:
        dist=np.abs(cp[:,None]-truth[None,:])/n if len(cp) else None
        values.append(max(dist.min(axis=0).max(),dist.min(axis=1).max()) if len(cp) else np.nan)
    H.append(values)
with plt.rc_context(PAPER_STYLE):
    fig,ax=paper_panel();ax.plot(saved['grid'],saved['tv'],color='#4477AA',lw=3.5*SCALE)
    ax.axvline(saved['gamma'],color='black',ls='--',lw=2*SCALE)
    ax.set(xlabel=r'$\gamma$',ylabel='Count-distribution TV');export_plot(fig,'M3_gamma_calibration')
    for groups,ylabel,name in [(H,r'Hausdorff dist. / $n$','location'),
                             (counts,'No. change points','counts'),
                             ([c+1 for c in counts],'Selected number of signals','signals')]:
        simple_summary(groups,[0,1],'',ylabel,'M3_'+name+'_box',colors=colors,labels=labels)
    histogram_grid([[np.concatenate(cp)/n for cp in cps]],labels,colors,truth/n,'M3_histogram',
                   density_label_x=.015)
    fig,ax=paper_panel(full=True,height=1.5)
    fig.subplots_adjust(left=.11,right=.985,bottom=.27,top=.80)
    correct=[int(np.sum(c==len(truth))) for c in counts]
    bars=ax.bar([0,1],correct,width=.62,color=colors,edgecolor='black',lw=SCALE)
    ax.set_xticks([0,1],labels);ax.set(xlim=(-.6,1.6),ylim=(0,115),ylabel='Correct trials')
    ax.set_yticks([0,50,100])
    for bar,count in zip(bars,correct):
        ax.text(bar.get_x()+bar.get_width()/2,count+2.5,f'{count}/100',ha='center',va='bottom',fontsize=8.5)
    export_plot(fig,'M3_correct_count_frequency')
print('Frozen gamma:',saved['gamma'],'; correct-count trials:',correct)
''')])

MI_SECTIONS=[]
for case,mode,folder in [('A','rank3','results/myocardial_final_redo_e3d965cc4b9b'),
                        ('B','whole','results/myocardial_final_final_v1/whole'),
                        ('C','segmented','results/myocardial_final_final_v1/segmented')]:
    MI_SECTIONS.append(section(f'Case {case}: {mode} — posterior and sensitivity',f"root=Path({folder!r})\nmode={mode!r}\n"+r'''
baseline=load_saved(root/'scenario_100_100_q10_seed123.pkl')
ages=np.asarray(baseline['ages']);prob=np.asarray(baseline['conditional']);finite=np.isfinite(ages)&np.isfinite(prob)
assert np.isclose(prob[finite].sum(),1)
print(mode,'1024 integration draws; P(1 chgpt) =',baseline['presence'])
with plt.rc_context(PAPER_STYLE):
    fig,ax=plt.subplots(figsize=(PAPER_WIDTH,1.18))
    fig.subplots_adjust(left=.11,right=.985,bottom=.39,top=.96)
    markers,stems,_=ax.stem(ages[finite],prob[finite],linefmt='k-',markerfmt='ko',basefmt=' ')
    plt.setp(stems,lw=1.05,alpha=.8);plt.setp(markers,markersize=2.3,markeredgewidth=0)
    ax.set(xlim=(26,95.1),xlabel='Age (years)');ax.set_xticks([26,40,50,60,75,81,92])
    ax.set_yticks([0,max(prob[finite])],['0',f'{max(prob[finite]):.2f}'])
    ax.grid(axis='y',ls=':',color='gray',alpha=.4)
    export_plot(fig,mode+'_conditional_posterior')

    levels=[80,90,100,110,120]
    scenarios={(u,v):load_saved(root/f'scenario_{u}_{v}_q10_seed123.pkl') for u in levels for v in levels}
    maximum=max(np.asarray(s['conditional'])[finite].max() for s in scenarios.values())
    fig,axes=plt.subplots(5,5,figsize=(PAPER_WIDTH,5.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.11,right=.93,bottom=.10,top=.93,wspace=.08,hspace=.12)
    for i,u in enumerate(levels):
        for j,v in enumerate(levels):
            ax=axes[i,j];values=np.asarray(scenarios[u,v]['conditional'])
            assert np.isclose(values[finite].sum(),1)
            ax.vlines(ages[finite],0,values[finite],color='black',lw=.6,alpha=.8)
            ax.plot(ages[finite],values[finite],'ko',ms=1.3)
            ax.set(xlim=(26,95.1),ylim=(0,1.08*maximum))
            ax.set_xticks([26,81]);ax.set_yticks([0,maximum],['0',f'{maximum:.2f}'])
            if i==0:ax.set_title(f'{v-100:+d}'+r'\%',pad=5)
            if j==4:ax.text(1.08,.5,f'{u-100:+d}'+r'\%',transform=ax.transAxes,rotation=90,va='center')
    fig.text(.5,.98,r'$\lambda_2$ perturbation',ha='center',va='top',fontsize=10.5)
    fig.text(.99,.5,r'$\lambda_1$ perturbation',ha='right',va='center',rotation=90,fontsize=10.5)
    fig.text(.5,.02,'Age (years)',ha='center',fontsize=10.5)
    fig.text(.015,.5,'Conditional posterior probability',rotation=90,va='center',fontsize=10.5)
    export_plot(fig,mode+'_sensitivity','Eigenvalue perturbations; all 25 saved scenarios, 1024 integration draws.')
'''))
PLAN['Myocardial Infarction Experiment - final_final.ipynb']=('Myocardial',MI_SECTIONS)

PLAN['FRED MD Final.ipynb']=('FRED',[
 section('Recorded final selections and response series',r'''
import pandas as pd
from matplotlib.dates import YearLocator, DateFormatter
fred=load_saved('FRED-MD data/fredmd_data.pkl')
dates=pd.to_datetime(np.asarray(fred['date']).astype(str));y=np.asarray(fred['y'])
# These are the completed result tables already displayed in cells 2 and 4 of
# this final notebook, not a new fit. The snapshot is explicit for standalone
# replotting; update it if intentionally replacing the underlying experiment.
recorded=[('WERM+L1',['1980-09','2009-06'],'black'),
          ('WERM (unpenalized CV)',[],'#4477AA'),
          ('WERM (original elbow)',['1982-01','2008-09'],'#AA3377')]
display(pd.DataFrame([{'Method':name,'No. change points':len(cp),
                      'Change-point dates':'; '.join(cp)} for name,cp,color in recorded]))
print('Date selections: completed final-notebook table snapshot; response: fredmd_data.pkl.')
with plt.rc_context(PAPER_STYLE):
    for j,(name,cp,color) in enumerate(recorded):
        fig,ax=plt.subplots(figsize=(PAPER_WIDTH,1.6))
        fig.subplots_adjust(left=.11,right=.985,bottom=.30,top=.96)
        ax.plot(dates,y,color='#555555',lw=.65)
        for date in cp:ax.axvline(pd.Timestamp(date),color=color,lw=1.5,ls='--')
        ax.set(xlabel='Year',ylabel='Response')
        ax.xaxis.set_major_locator(YearLocator(10));ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.yaxis.set_major_locator(MaxNLocator(3));ax.set_xlim(dates[0],dates[-1])
        export_plot(fig,f'FRED_response_{j+1}',name+'; selected dates: '+(', '.join(cp) if cp else 'none'))
    fig,ax=paper_panel(full=True)
    for j,(name,cp,color) in enumerate(recorded):
        ax.hlines(j,dates[0],dates[-1],color='#cccccc',lw=.7)
        if cp:ax.scatter(pd.to_datetime(cp),[j]*len(cp),marker='D',color=color,s=24)
    ax.set_yticks(range(3),[r'\textbf{WERM+L1}','Unpen. CV','Elbow'])
    ax.set(xlabel='Year',ylim=(-.6,2.6),xlim=(dates[0],dates[-1]))
    ax.xaxis.set_major_locator(YearLocator(10));ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    fig.subplots_adjust(left=.20)
    export_plot(fig,'FRED_selection_comparison')
''')])

def build(name):
    slug,sections=PLAN[name]
    path=ROOT/name
    original=nbformat.read(path,as_version=4)
    preserved=[copy.deepcopy(c) for c in original.cells if TAG not in c.get('metadata',{}).get('tags',[])]
    cells=[]
    def add(cell):
        cell.metadata['tags']=[TAG];cells.append(cell)
    add(nbformat.v4.new_markdown_cell('## Updated consistent plots\n\n'
        'Plotting reference: **Canonical M1 tighter-layout plots (alpha = 0.3)** in '
        '`M1_upsample_final.ipynb`. Error-bar plots use its 5 by 4 inch canvas, '
        '18 pt axis labels and 14 pt ticks; among Hausdorff, count and runtime comparisons, '
        'only Runtime displays a legend. These cells load saved results only; '
        'the original experiment and plotting cells above are unchanged. Run the setup below, '
        'then the desired plotting cell. PDFs and PNGs are saved under '
        f'`results/consistent_plots_20261102/{slug}/`.\n\n'
        'Line summaries retain their original statistic (quartile-distance bars, or mean ± SE '
        'where stated); boxplots use 1.5-IQR whiskers and faint deterministic trial jitter. '
        'Histograms follow canonical M1: 12 data-range bins, five-fold bandwidth-selected Gaussian KDE, '
        'grey overlay, truth-position ticks, and density range 0–18. These histogram display choices '
        'can differ from the older plots; the samples do not change. Empty estimates are omitted from '
        'conditional localization and pooled location densities, not assigned zero error. '
        'Histograms retain every selected setting and method in one grid, with no three-row or '
        'three-column cap: 2.5 inches per column and 2.2 inches per row, matching the reference. '
        'Other diagnostic plots retain their compact presentation.'))
    add(nbformat.v4.new_code_cell(HELPERS+f"\nOUTPUT_DIR=Path('results/consistent_plots_20261102/{slug}')\n"))
    for title,code in sections:
        add(nbformat.v4.new_markdown_cell('### '+title))
        add(nbformat.v4.new_code_cell(code))
    add(nbformat.v4.new_code_cell('verify_sources()'))
    # Execute only the new cells in a fresh kernel. The old experiment cells
    # cannot be run by this client, because they are not in this notebook object.
    mini=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    print('Rendering',name,flush=True)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0]=sys.executable
    manager.transport='ipc'
    socket_dir=tempfile.mkdtemp(prefix='consistent-plots-kernel-')
    manager.ip=str(Path(socket_dir)/'socket')
    client=NotebookClient(mini,timeout=1200,resources={'metadata':{'path':str(ROOT)}},
                          kernel_name='python3',km=manager)
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        shutil.rmtree(socket_dir,ignore_errors=True)
    assert all(o.output_type!='error' for c in mini.cells for o in c.get('outputs',[]))
    # Re-read before saving so unrelated external edits cannot be overwritten.
    current=nbformat.read(path,as_version=4)
    current_preserved=[c for c in current.cells if TAG not in c.get('metadata',{}).get('tags',[])]
    assert current_preserved==preserved,'Notebook changed during rendering: '+name
    current.cells=current_preserved+mini.cells
    nbformat.validate(current)
    temp=path.with_suffix('.consistent-tmp.ipynb')
    nbformat.write(current,temp);temp.replace(path)
    check=nbformat.read(path,as_version=4)
    assert check.cells[:len(preserved)]==preserved
    print('Saved',name,'new plots:',sum('image/png' in o.get('data',{}) for c in mini.cells for o in c.get('outputs',[])),flush=True)

if __name__=='__main__':
    # M2 is deliberately absent, and protected against accidental modifications.
    m2=ROOT/'M2_upsample_final.ipynb';m2_hash=hashlib.sha256(m2.read_bytes()).hexdigest()
    names=sys.argv[1:] or list(PLAN)
    for name in names:build(name)
    if 'M2_upsample_final.ipynb' not in names:
        assert hashlib.sha256(m2.read_bytes()).hexdigest()==m2_hash
