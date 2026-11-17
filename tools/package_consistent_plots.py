"""Validate rendered cells/assets and package their vector PDFs for Overleaf."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/consistent_plots_20261102'
TAG='updated-consistent-plots-20261102'
NAMES={
    'M2':'M2_upsample_final.ipynb',
    'M1':'M1_upsample_final.ipynb', 'M3':'M3_upsample_final.ipynb',
    'Myocardial':'Myocardial Infarction Experiment - final_final.ipynb',
    'E_L_upper_bound':'E_L_upper_bound.ipynb',
    'E_PRIOR':'E_PRIOR_misspecified_prior.ipynb', 'E_search_penalty_M3':'E_search_penalty.ipynb',
}
def included(slug,figure):
    if slug=='M3' and figure['name'].startswith('M3_delta6_'):
        return False
    return slug!='Myocardial' or figure['name'].startswith('rank3_')

report={}
for slug,name in NAMES.items():
    notebook=nbformat.read(ROOT/name,as_version=4)
    cells=[c for c in notebook.cells if TAG in c.metadata.get('tags',[])]
    assert cells and sum(c.cell_type=='markdown' and c.source.startswith('## Updated consistent plots') for c in cells)==1
    assert all(c.execution_count is not None for c in cells if c.cell_type=='code')
    outputs=[o for c in cells for o in c.get('outputs',[])]
    assert not [o for o in outputs if o.output_type=='error']
    images=sum('image/png' in o.get('data',{}) for o in outputs)
    manifest=json.loads((OUT/slug/'manifest.json').read_text())
    assert images==len(manifest['figures']),(slug,images,len(manifest['figures']))
    for path,digest in manifest['sources'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for figure in manifest['figures']:
        if not included(slug,figure):
            continue
        if slug in ('M1','M2','M3'):
            legend_metric='size' if slug=='M3' else 'runtime'
            expected=int(figure['name'].endswith((f'_{legend_metric}_line',f'_{legend_metric}_box')))
            assert figure['legends']==expected,(slug,figure['name'],'noncanonical legend placement')
        pdf=OUT/slug/(figure['name']+'.pdf')
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        assert re.search(r'Pages:\s+1\b',info),pdf
        dims=re.search(r'Page size:\s+([\d.]+) x ([\d.]+)',info)
        assert dims
        for actual,expected in zip(map(float,dims.groups()),figure['size_inches']):
            assert abs(actual-72*expected)<.02,(pdf,actual,expected)
        fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        assert 'Type 3' not in fonts,pdf
    report[slug]={'notebook':name,'embedded_plots':images,
                  'included_plots':sum(included(slug,f) for f in manifest['figures']),
                  'source_files':len(manifest['sources'])}

tex=r'''\documentclass[12pt]{article}
\usepackage[paperwidth=188.88mm,paperheight=246mm,textwidth=427.64493pt,textheight=596.5pt,centering]{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[T1]{fontenc}
\captionsetup{font=small}
\begin{document}
\begin{figure}[t]
\centering
\begin{subfigure}[t]{.49\linewidth}
\includegraphics[width=\linewidth]{M3/M3_delta3_haus_box.pdf}
\caption{Location error.}
\end{subfigure}\hfill
\begin{subfigure}[t]{.49\linewidth}
\includegraphics[width=\linewidth]{M3/M3_delta3_size_line.pdf}
\caption{Selected counts.}
\end{subfigure}
\caption{M3, $\delta=3$, $\alpha=0.1$: saved 100-trial comparison.
Boxplots show the median and interquartile range with 1.5-IQR whiskers
and faint trial points. Count lines show means with the canonical
absolute distances to the first and third quartiles.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{E_PRIOR/prior_estimated_locations.pdf}
\caption{Estimated locations under bad, neutral and good priors.
The canonical histogram display uses 12 bins and a Gaussian KDE;
dashed lines mark truth. Density peaks above 18 are clipped.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{Myocardial/rank3_conditional_posterior.pdf}
\caption{Case A myocardial posterior conditional on one change point.}
\end{figure}
\end{document}
'''
(OUT/'include_examples.tex').write_text(tex)
subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','include_examples.tex'],
    cwd=OUT,check=True,stdout=(OUT/'include_examples.stdout').open('w'))
notes='''UPDATED CONSISTENT PLOTS — 2026-11-02

These PDFs were rendered from saved results. Experiments were not rerun.
New rendered cells appear under "Updated consistent plots" in the final
notebooks listed in validation.json. Original cells are untouched. M2 now includes
the completed fresh four-method comparison (gamma=6); four charcoal R failures
remain missing values, distinct from valid empty estimates.

Upload the folders in this archive to Overleaf. include_examples.tex gives a
compiled example. Comparison error-bar plots follow the canonical M1 tighter-layout reference:
5 by 4 inch canvases, 18 pt axis labels, 14 pt ticks, and a legend only on Runtime.
Histograms retain all selected settings and methods in one grid with no three-row
or three-column cap (2.5 inches per column, 2.2 inches per row). Canonical exports
use the reference's tight crop. Other diagnostics retain their compact layout.
Scale whole figures proportionally when including them in the manuscript.

Computer Modern and vector PDF throughout.
Method colors/markers follow the approved specimen. Bold WERM labels only.
Histogram display follows canonical M1: 12 data-range bins, five-fold bandwidth
selection for Gaussian KDE, gray overlay, fixed 0–18 density scale. The notebook
reports peaks that exceed this scale. Individual data and model fits are unchanged.
Line uncertainty summaries retain their original definition: quartile-distance
bars for M1/M3/upper-L, mean +/- SE for fixed-L and E-PRIOR. These are not confidence
intervals. Conditional localization excludes empty change-point estimates.

FRED, myocardial cases B/C, fixed-L and M3 delta=6 are excluded from this handoff at the user's request.
Their existing notebook results remain intact.
Only the final independent M3 comparison is updated in E_search_penalty.
Only myocardial case A is included; sensitivity uses all 25 saved scenarios, 1024 draws.

To regenerate: run the new setup cell, then the desired plot cells in a notebook.
No earlier experiment cell needs to run. Each notebook's manifest records saved
input hashes and figure geometry. Helpers are embedded visibly in each notebook.
'''
(OUT/'README.txt').write_text(notes)
(OUT/'validation.json').write_text(json.dumps(report,indent=2))
with zipfile.ZipFile(OUT/'overleaf_consistent_plots.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for slug in NAMES:
        manifest=json.loads((OUT/slug/'manifest.json').read_text())
        for figure in manifest['figures']:
            if not included(slug,figure):
                continue
            path=OUT/slug/(figure['name']+'.pdf')
            archive.write(path,str(path.relative_to(OUT)))
    for filename in ('include_examples.tex','include_examples.pdf','README.txt','validation.json'):
        archive.write(OUT/filename,filename)
print(json.dumps(report,indent=2))
print('Total included validated plots:',sum(v['included_plots'] for v in report.values()))
