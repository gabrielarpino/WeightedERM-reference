UPDATED CONSISTENT PLOTS — 2026-11-02

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
