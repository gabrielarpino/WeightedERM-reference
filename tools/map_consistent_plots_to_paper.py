"""Read-only manuscript audit; write figure placement recommendations locally."""
from pathlib import Path
import csv,json,subprocess,hashlib

root=Path(__file__).resolve().parents[1]
out=root/'results/consistent_plots_20261102'
paper=root.parent/'WERM-paper'
tex=paper/'JRSSB-revision-rebuttal/JRSSB_revision.tex'
lines=tex.read_text().splitlines()
commit=subprocess.check_output(['git','-C',str(paper),'rev-parse','--short','HEAD'],text=True).strip()

def locate(label):
    return next((i+1 for i,line in enumerate(lines) if r'\label{'+label+'}' in line and not line.lstrip().startswith('%')),None)

def placement(slug,name):
    status='optional appendix addition'
    target='sec:additional-experiments-and-examples'
    note='No current image slot; keep as supplementary material unless needed.'
    if slug=='M1':
        if name.startswith('M1a_'):
            target='fig:M1a'
            if name.endswith(('haus_line','runtime_line')):status='replace existing panel'
        elif name.startswith('M1b_alpha03_top3'):
            target='sec:lin-model-sparse-signals';note='Optional restricted three-method supplement; do not replace the full comparison.'
        elif name.startswith('M1b_alpha03_'):
            target='fig:M1b-hist' if 'hist' in name else 'fig:M1b'
            if 'hist' in name or name.endswith(('haus_line','size_line','runtime_line')):status='replace existing panel'
        elif name.startswith('M1b_dense_student'):
            target='fig:dense-gaussian-comparison'
            if name.endswith(('haus_box','size_line')):status='replace existing panel'
            note='Current subsection describes Gaussian SIGNALS but Student-t NOISE. This is its matching experiment.'
        else:
            target='sec:dense-gaussian-comparison';note='Add separate Gaussian-NOISE/least-squares comparison after existing Student-t/Huber figure; do not substitute under existing prose.'
    elif slug=='M2':
        if name.startswith('M2_original_'):
            target='fig:M2-hist' if 'hist' in name else 'fig:M2'
            if 'hist' in name or name.endswith(('haus_line','size_line','runtime_line')):
                status='replace existing panel, original three-method experiment'
            note='Original canonical M2 artifact: McScan, charcoal and unpenalized WERM 10-fold CV. Separate from the fresh four-method run; do not mix their fits or runtimes.'
        else:
            target='sec:investigating-count-selection'
            if 'hist' in name or name.endswith(('haus_line','size_line','runtime_line')):
                status='proposed separate M2 appendix figure, no existing slot'
            note='Recommend a new M2 comparison paragraph/figure after the existing M3 count-selection comparison, with proposed labels fig:M2-count-selection-comparison and fig:M2-count-selection-hist. Fresh four-method experiment includes WERM+P gamma=6; do not overwrite the original M2 or M3 figures. Four charcoal R failures are excluded; charcoal stops at delta<=9.'
    elif slug=='M3':
        target='fig:M3b' if 'delta3' in name else 'sec:logistic-model-sparse-signals-exp'
        if 'delta3' in name and name.endswith(('haus_box','size_line','runtime_box')):status='recommended main-text replacement, resolve mismatch first'
        note='Main prose specifies delta=3 but existing figure paths use delta6. Recommend delta3 in main; delta6 is excluded from the current handoff. Histograms have no active main-text slot.'
    elif slug=='FRED':
        target='sec:FRED';status='new figure, no current slot'
        note='Primary response_1 or selection_comparison after reported change dates. Other response plots optional. Dates are recorded notebook table outputs; not new fits.'
    elif slug=='Myocardial':
        if name=='rank3_conditional_posterior':
            target='fig:myocardial_chgpts';status='replace existing figure'
        elif name=='rank3_sensitivity':
            target='tab:myocardial-rho-sensitivity';status='new figure after existing table'
        else:
            target='sec:myocardial-sensitivity';note='Whole/segmented Gram alternatives need their own explanation and tables; not replacements for the rank-3 posterior or sensitivity table.'
    elif slug=='E_L_fixed':
        target='sec:sensitivity-number-of-change-points';note='No active slot: fixed working L is a different estimator/experiment from upper-bound CV. Optional separate fixed-L subsection; never replace the upper-bound panels.'
    elif slug=='E_L_upper_bound':
        target='fig:E-L-hist' if 'hist' in name else 'fig:E-L-location'
        if name in ('upper_L_count_line','upper_L_haus_box','upper_L_histograms'):status='replace existing panel'
        else:note='Optional diagnostics alongside the upper-bound subsection.'
    elif slug=='E_PRIOR':
        target={'pooled_prior':'fig:E-PRIOR-priors','signal_marginals':'fig:E-PRIOR-priors',
                'prior_location_box':'fig:E-PRIOR-location','prior_estimated_locations':'fig:E-PRIOR-hist'}.get(name,'fig:E-PRIOR-priors')
        if name in ('pooled_prior','signal_marginals','prior_location_box','prior_estimated_locations'):status='replace existing panel'
        else:note='Optional additional prior diagnostic; joint prior is distinct from pooled-location prior.'
    elif slug=='E_search_penalty_M3':
        target='fig:M3-count-selection-procedures'
        if name in ('M3_histogram','M3_correct_count_frequency'):status='replace existing panel'
        else:note='Optional calibration/location/count diagnostics; calibration plot requires restoring the currently commented calibration explanation.'
    if status.startswith('replace') and note=='No current image slot; keep as supplementary material unless needed.':
        note='Matching existing figure slot; use this experiment\'s updated consistent export.'
    return status,target,note

rows=[]
for manifest_path in sorted(out.glob('*/manifest.json')):
    slug=manifest_path.parent.name
    if slug in ('FRED','E_L_fixed'):
        continue
    for fig in json.loads(manifest_path.read_text())['figures']:
        if slug=='M3' and fig['name'].startswith('M3_delta6_'):
            continue
        if slug=='Myocardial' and not fig['name'].startswith('rank3_'):
            continue
        status,target,note=placement(slug,fig['name'])
        asset=f"JRSSB-revision-rebuttal/figures/consistent_20261103/{slug}/{fig['name']}.pdf"
        uploaded=(asset in tex.read_text() and (paper/asset).exists() and
                  hashlib.sha256((paper/asset).read_bytes()).digest()==
                  hashlib.sha256((manifest_path.parent/(fig['name']+'.pdf')).read_bytes()).digest())
        if uploaded:
            status='updated in manuscript; awaiting review'
            note='Included consistent PDF; experimental description and caption updated in the authorized appendix sections.'
            if slug=='M2':
                target='fig:M2-count-selection-hist' if 'hist' in fig['name'] else 'fig:M2-count-selection-comparison'
        w,h=fig['size_inches']
        rows.append(dict(pdf=f"{slug}/{fig['name']}.pdf",disposition=status,
            manuscript_asset=asset if uploaded else '',
            latex_target=target,tex_line=locate(target),width_inches=round(w,3),height_inches=round(h,3),
            notes=note))
with (out/'JRSSB_v2_placement_map.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

text=f'''FIGURE AUDIT AND PROPOSED PLACEMENTS — 2026-11-03
Paper: WERM-paper/JRSSB-revision-rebuttal/JRSSB_revision.tex, Overleaf commit {commit}.
Only the authorized E-L upper-bound, E-PRIOR, E-search genuine M3 and fresh
four-method M2 appendix sections/assets were updated. Line numbers refer to this commit.
JRSSB_v2_placement_map.csv inventories every active exported figure ({len(rows)}).
It separates existing replacements from optional additions: not all notebook
diagnostic plots should be put into the paper.
FRED, myocardial cases B/C, fixed-L and M3 delta=6 are excluded from this handoff.
Their notebook results remain intact; only myocardial case A is included here.

PRIORITY EXISTING SLOTS
M1a: fig:M1a (line {locate('fig:M1a')}): fixed-count location and runtime lines.
M1b alpha .3: fig:M1b (line {locate('fig:M1b')}): location, count, runtime lines;
  fig:M1b-hist: full-method histogram. Top-three variants remain supplementary.
M2 ORIGINAL THREE-METHOD: fig:M2 (line {locate('fig:M2')}): M2_original_haus_line,
  M2_original_size_line, M2_original_runtime_line. fig:M2-hist: M2_original_hist.
  Original boxplot alternatives are optional supplementary plots.
M2 FRESH FOUR-METHOD: NEW M2 paragraph and figures added in
  sec:investigating-count-selection (line {locate('sec:investigating-count-selection')}),
  after the existing M3 comparison. Use M2_fresh_haus_line, M2_fresh_size_line,
  M2_fresh_runtime_line and M2_fresh_hist. New labels:
  fig:M2-count-selection-comparison and fig:M2-count-selection-hist.
  This appendix update is awaiting review. Keep both M2 experiments
  distinct; the fresh comparison is not the original three-method experiment.
M3: fig:M3b (line {locate('fig:M3b')}): location box, count line, runtime box.
  Prose says delta=3, existing figure paths say delta6. Resolve before replacing;
  recommended main result is delta3 to match prose. Delta6 is excluded for now.
Myocardial: fig:myocardial_chgpts (line {locate('fig:myocardial_chgpts')}): rank3 posterior.
  Add rank3 sensitivity grid after tab:myocardial-rho-sensitivity.
Dense M1: fig:dense-gaussian-comparison (line {locate('fig:dense-gaussian-comparison')}):
  Student-t/Huber location box and count line. Add Gaussian-noise/LS separately.
Upper-bound L: fig:E-L-location (line {locate('fig:E-L-location')}): count line + location box;
  fig:E-L-hist: upper_L_histograms. Fixed-L results are excluded for now.
E-search M3: fig:M3-count-selection-procedures (line {locate('fig:M3-count-selection-procedures')}):
  M3_histogram and M3_correct_count_frequency only; other panels are diagnostics.
E-PRIOR: fig:E-PRIOR-priors: pooled prior above signal marginals;
  fig:E-PRIOR-location: location box; fig:E-PRIOR-hist: estimated-location histogram.

AUDIT FINDINGS TO ADDRESS WHEN INSERTING
1. The FRESH FOUR-METHOD M2 experiment has all intended trial records: 1000 each for McScan, WERM, WERM+P and
   800 attempted charcoal fits through delta=8.5. Four charcoal R errors were
   stored as empty estimates: delta5.5 trials54/82 and delta8.5 trials41/81.
   The new plotting cells exclude these four from metric summaries and state
   the exclusion. Raw artifacts and old notebook plots are unchanged. Valid
   zero-change estimates remain zero counts, with undefined location errors.
2. The latest canonical overrides retain 5x4-inch lineplot canvases and
   2.5-inch-per-column histogram grids with 18pt labels and tight cropping.
   Compact boxes/diagnostics use 10.5pt labels and fixed canvases. They render
   correctly but are NOT guaranteed identical printed sizes under arbitrary
   includegraphics widths. The CSV gives actual exported PDF dimensions.
   Printed label size = source label size * inserted width / exported width.
   Current three-panel .32-textwidth rows shrink the compact panels and require
   re-layout (prefer two panels per row) or a dedicated three-panel export.
   Large canonical histogram grids likewise need explicit printed-size review.
   Do not claim globally identical typography from a successful PDF compile.
3. E-search now stacks the histogram at half text width above the full-width
   count bar, retaining comparable printed label sizes despite their different
   source typography. Fresh M2 uses two panels above the centered runtime panel.
   Review these layouts and the small six-column upper-L histogram in Overleaf.
4. M3 prose delta3 vs figure delta6 is a genuine design mismatch. The notebook
   WERM CV uses minimum separation n/10, whereas the main prose says n/20;
   GLMBSA uses kmax=L-2=2, whereas prose says maximum L-1=3. Clarify parameter
   meanings and actual protocol when updating this section.
5. FRED, myocardial cases B/C, fixed-L and M3 delta6 are excluded from this handoff.
6. Dense 'Gaussian signals' currently has Student-t noise. A separate Gaussian
   noise/least-squares experiment exists; substituting it silently would mislabel
   both the noise distribution and loss.
7. Histogram captions still mentioning 30 trials must say 100 where appropriate.
   Explain the canonical 12-bin/KDE visualization and fixed 0-18 density limit;
   peaks above the limit are reported by the notebook. Some older histogram
   exports used different bins or no KDE.
8. Error bars use mean +/- absolute distances to Q1/Q3, not confidence intervals;
   if a mean lies outside the IQR they are not literally Q1-to-Q3 endpoints.
9. E-search gamma=6 was selected using saved calibration trials. The updated
   subsection restores that provenance and distinguishes the fresh M2 comparison,
   which reuses gamma=6 without recalibration.

Validation: all active new cells execute without saved error outputs; image
counts match manifests; input hashes are unchanged; every active PDF is a single
page with dimensions matching its manifest and no Type-3 fonts. See validation.json.
The bundle uses active manifests, so obsolete exports on disk are not included.
'''
(out/'JRSSB_v2_placement_notes.txt').write_text(text)
print('Mapped',len(rows),'figures; paper commit',commit)
