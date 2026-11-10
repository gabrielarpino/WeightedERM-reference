"""Canonical compact plots for the saved unknown-count WERM/AMP comparison.

Only completed checkpoints are read. No estimator fitting or simulation is
performed. The geometry and plotting conventions match the compact known-count
figures and the final M1/M2/M3 notebooks.
"""
from pathlib import Path
import csv
import hashlib
import json
from math import comb, log

import numpy as np

from tools.werm_amp_compact_plots import _export, _sha256
from tools.werm_amp_experiment import COVARIANCE_TITLES, METHOD_STYLES, WERM_P_VERSION, load_results


DEFAULT_ROOT = (Path(__file__).resolve().parents[1]
                / 'results/WERM_AMP_comparison/7b64ccb1c8c6')
COVARIANCES = ('identity', 'ar1', 'm3')
METHODS = ('WERM+CV', 'WERM+P', 'AMP')
DELTAS = (1.5, 3.5, 7.5, 11.5)
HISTOGRAM_DELTAS = (7.5, 11.5)
HISTOGRAM_DENSITY_LIMIT = 24.
TITLES = {**COVARIANCE_TITLES, 'm3': 'random orthogonal'}
LABELS = {'WERM+CV': r'\textbf{WERM+CV}',
          'WERM+P': r'\textbf{WERM+P}', 'AMP': 'AMP'}
FIVE_COLUMN_HISTOGRAM = (('ar1', 'WERM+CV'), ('ar1', 'WERM+P'),
                         ('identity', 'AMP'), ('ar1', 'AMP'), ('m3', 'AMP'))
PAPER_COVARIANCE_TITLES = {'identity': 'isotropic', 'ar1': 'AR(1)', 'm3': 'orthogonal'}


def _validated_groups(root, expected_truth=(.25, .50)):
    """Require the complete paired unknown-count study and validate its metrics."""
    expected_truth = np.asarray(expected_truth, dtype=float)
    if (expected_truth.shape != (2,) or not np.isfinite(expected_truth).all()
            or not np.all((expected_truth > 0) & (expected_truth < 1))
            or not np.all(np.diff(expected_truth) > 0)):
        raise ValueError('expected_truth must contain two increasing finite fractions in (0, 1)')
    root = Path(root)
    payload = load_results(root)
    spec = payload['specification']
    saved_truth = np.asarray(spec.get('unknown_truth'), dtype=float)
    if (saved_truth.shape != (2,)
            or not np.allclose(saved_truth, expected_truth, rtol=0, atol=1e-12)):
        raise ValueError(f'Expected unknown_truth={expected_truth.tolist()}; '
                         f'found {spec.get("unknown_truth")!r}')
    required = {'p': 200, 'trials': 100, 'max_segments': 4, 'amp_iterations': 10,
                'folds': 10, 'gamma': 6., 'whitening': False,
                'response_standardization': False, 'coefficient_loss': 'squared',
                'coefficient_penalty': 'none', 'cv_validation_loss': 'absolute',
                'werm_p_version': WERM_P_VERSION, 'penalty_search_loss': 'half squared'}
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise ValueError(f'Expected {key}={expected!r}; found {spec.get(key)!r}')
    if (tuple(spec['deltas']) != DELTAS
            or set(spec['covariances']) != set(COVARIANCES)
            or not np.isclose(spec['separation_fraction'], .1)
            or not np.isclose(spec['ar1_correlation'], .4)
            or 'uniform source label subsets:' not in spec['amp_configuration_prior']):
        raise ValueError('Expected the corrected n/10, unknown-count comparison')
    spec_digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    if spec_digest != root.name:
        raise ValueError('Saved specification does not match its run digest')

    groups = {(covariance, delta): []
              for covariance in COVARIANCES for delta in DELTAS}
    seen = set()
    hashes = {'specification.json': _sha256(root / 'specification.json')}
    for record in payload['records']:
        if record['study'] != 'unknown':
            continue
        key = (record['covariance'], record['delta'])
        trial_key = (*key, record['trial'])
        if key not in groups or trial_key in seen:
            raise ValueError(f'Unexpected or duplicate unknown-count trial: {trial_key}')
        seen.add(trial_key)
        if record['spec_digest'] != spec_digest:
            raise ValueError(f'Checkpoint belongs to a different run: {trial_key}')
        n = round(record['delta'] * spec['p'])
        truth = np.rint(n * np.asarray(spec['unknown_truth'])).astype(int)
        if (record['n'] != n or record['p'] != spec['p']
                or not np.array_equal(record['truth'], truth)):
            raise ValueError(f'Inconsistent checkpoint dimensions/truth: {trial_key}')
        if set(record['methods']) != set(METHODS):
            raise ValueError(f'Expected WERM+CV, WERM+P and AMP: {trial_key}')
        for method, fit in record['methods'].items():
            if fit.get('status') != 'ok':
                continue  # Failures are retained in the audit, never imputed.
            cp = np.asarray(fit['cp'])
            if (cp.ndim != 1 or not 0 <= len(cp) <= 3
                    or fit['count'] != len(cp) or fit['segments'] != len(cp) + 1
                    or bool(fit['empty']) != (len(cp) == 0)
                    or bool(fit['correct_count']) != (len(cp) == len(truth))
                    or not np.all((cp > 0) & (cp < n))
                    or not np.all(np.diff(cp) > 0)
                    or not np.array_equal(cp, cp.astype(int))):
                raise ValueError(f'Invalid count/location output: {trial_key}, {method}')
            if len(cp):
                distances = np.abs(cp[:, None] - truth[None, :]) / n
                haus = max(distances.min(axis=0).max(), distances.min(axis=1).max())
                if not np.isclose(fit['haus'], haus, rtol=1e-12, atol=1e-12):
                    raise ValueError(f'Incorrect normalized Hausdorff metric: {trial_key}, {method}')
            elif not np.isnan(fit['haus']):
                raise ValueError(f'Empty estimate must have undefined Hausdorff: {trial_key}, {method}')
            if method == 'WERM+P':
                expected_penalties = np.array([6*log(comb(n-1,k-1)) for k in range(1,5)])
                if (fit.get('coefficient_fit_scheme') != WERM_P_VERSION
                        or fit.get('search_loss') != 'half squared'
                        or not np.allclose(fit['candidate_penalties'], expected_penalties,
                                           rtol=1e-10, atol=1e-10)
                        or not np.allclose(fit['candidate_total'],
                                           fit['candidate_raw']+expected_penalties)
                        or fit['segments'] != int(np.argmin(fit['candidate_total']))+1
                        or not np.array_equal(cp, fit['candidate_cp'][fit['segments']-1])):
                    raise ValueError(f'Expected corrected single-fit WERM+P: {trial_key}')
            if method == 'AMP' and (fit.get('iterations_requested') != 10
                                    or fit.get('iterations_completed') != 10
                                    or fit.get('fixed_count') is not False
                                    or fit.get('design_transform') != 'none'):
                raise ValueError(f'Unexpected AMP settings: {trial_key}')
        groups[key].append(record)
        filename = (f'unknown__{record["covariance"]}__d{record["delta"]:g}'
                    f'__t{record["trial"]:03d}.pkl')
        hashes[filename] = _sha256(root / filename)

    for key, records in groups.items():
        if len(records) != 100 or {r['trial'] for r in records} != set(range(100)):
            raise ValueError(f'Expected trials 0..99 for {key}; found {len(records)}')
        records.sort(key=lambda record: record['trial'])
    for delta in DELTAS:
        for trial in range(100):
            if len({groups[covariance, delta][trial]['latent_hash']
                    for covariance in COVARIANCES}) != 1:
                raise ValueError(f'Covariance trials do not share latent draws: {delta}, {trial}')
    return spec, groups, hashes


def _summary(groups):
    """Count all successful estimates; condition Hausdorff on nonempty ones."""
    rows = []
    for covariance in COVARIANCES:
        for delta in DELTAS:
            for method in METHODS:
                fits = [r['methods'][method] for r in groups[covariance, delta]]
                successful = [f for f in fits if f['status'] == 'ok']
                nonempty = [f for f in successful if len(f['cp'])]
                counts = np.asarray([f['count'] for f in successful], dtype=float)
                haus = np.asarray([f['haus'] for f in nonempty], dtype=float)
                if not np.isfinite(haus).all():
                    raise ValueError(f'Invalid nonempty Hausdorff value: {covariance}, {delta}, {method}')
                row = dict(covariance=covariance, delta=delta, method=method,
                    trials=len(fits), successes=len(successful), failures=len(fits)-len(successful),
                    nonempty_successes=len(nonempty), empty_estimates=len(successful)-len(nonempty),
                    failure_rate=(len(fits)-len(successful))/len(fits),
                    empty_rate=(len(successful)-len(nonempty))/len(fits),
                    correct_count_rate=sum(f['correct_count'] for f in successful)/len(fits))
                for name, values in (('haus', haus), ('count', counts)):
                    if not len(values):
                        # All-empty estimates have no conditional Hausdorff
                        # value. Keep nulls in the audit and gaps in the curve.
                        row.update({f'{name}_{stat}': None
                                    for stat in ('mean', 'q25', 'median', 'q75')})
                        continue
                    q25, median, q75 = np.quantile(values, [.25, .5, .75])
                    row.update({f'{name}_mean': float(values.mean()),
                                f'{name}_q25': float(q25), f'{name}_median': float(median),
                                f'{name}_q75': float(q75)})
                rows.append(row)
    return rows


def _metric_plots(summary, output_dir, artifacts, *, metrics=('haus', 'count'), titles=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    upper = 1.08 * max(row[key] for row in summary
                      for key in ('haus_mean', 'haus_q75') if row[key] is not None)
    titles = TITLES if titles is None else titles
    paths = []
    for metric in metrics:
        for index, covariance in enumerate(COVARIANCES):
            fig, ax = plt.subplots(figsize=(5., 4.))
            for method in METHODS:
                rows = [row for row in summary
                        if row['covariance'] == covariance and row['method'] == method]
                x = [row['delta'] for row in rows]
                q25 = np.asarray([row[f'{metric}_q25'] for row in rows], dtype=float)
                q75 = np.asarray([row[f'{metric}_q75'] for row in rows], dtype=float)
                means = np.asarray([row[f'{metric}_mean'] for row in rows], dtype=float)
                color, marker, line = METHOD_STYLES[method]
                # Quartile endpoints remain at their actual positions, including
                # when a discrete/skewed sample has its mean outside the IQR.
                ax.vlines(x, q25, q75, color=color, linewidth=3.5, alpha=.8)
                for quantile in (q25, q75):
                    ax.plot(x, quantile, ls='none', marker='_', markersize=8,
                            markeredgewidth=2, color=color, alpha=.8)
                ax.plot(x, means, label=LABELS[method],
                        color=color, marker=marker, ls=line, linewidth=3.5,
                        markersize=8, markeredgewidth=2, markeredgecolor='white', alpha=.8)
            ax.set(xlabel=r'$\delta=n/p$', title=titles[covariance])
            ax.set_xticks(DELTAS, [f'{value:g}' for value in DELTAS])
            ax.tick_params(axis='both', which='major', width=1.5, length=6, labelsize=14)
            if metric == 'haus':
                ax.set(ylabel=r'Hausdorff dist. / $n$', ylim=(-.03*upper, max(.001, upper)))
                ax.yaxis.set_major_locator(MaxNLocator(5))
            else:
                ax.set(ylabel='No. change points', ylim=(-.12, 3.2), yticks=[0, 1, 2, 3])
                ax.axhline(2, color='gray', lw=1.3, ls=':', zorder=0)
                if index == 0:
                    ax.legend(loc='lower right', fancybox=True, framealpha=.4)
            fig.tight_layout()
            paths.append(_export(fig, output_dir,
                f'unknown_gap10_T10_{covariance}_{metric}', artifacts))
    return paths


def _histogram(groups, truth, output_dir, artifacts, *, columns=None,
               stem='unknown_gap10_T10_hist'):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    grouped_covariances = columns is None
    columns = tuple((c, m) for c in COVARIANCES for m in METHODS) if columns is None else tuple(columns)
    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(2.5*ncols, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.04 if grouped_covariances else .06,
                        right=.975 if grouped_covariances else .96, bottom=.18, top=.80,
                        wspace=.05, hspace=.08)
    audit = []
    for row, delta in enumerate(HISTOGRAM_DELTAS):
        for column, (covariance, method) in enumerate(columns):
            ax = axes[row, column]
            records = groups[covariance, delta]
            fits = [record['methods'][method] for record in records]
            values = np.concatenate([np.asarray(record['methods'][method]['cp']) / record['n']
                for record in records if record['methods'][method]['status'] == 'ok'])
            if len(values) < 5:
                raise ValueError(f'Insufficient locations for histogram/KDE: {covariance}, {delta}, {method}')
            hist, edges, _ = ax.hist(values, bins=12, density=True, alpha=.8,
                color=METHOD_STYLES[method][0], edgecolor='white', linewidth=1.)
            if not np.isclose(np.sum(hist * np.diff(edges)), 1.):
                raise ValueError('Histogram density does not integrate to one')
            cv = GridSearchCV(KernelDensity(kernel='gaussian'),
                {'bandwidth': np.logspace(-2, 0, 50)}, cv=5, n_jobs=1)
            cv.fit(values.reshape(-1, 1))
            x = np.linspace(0, 1, 200)
            density = np.exp(cv.best_estimator_.score_samples(x.reshape(-1, 1)))
            ax.plot(x, density, color='#555555', lw=1., alpha=.8, zorder=2)
            ax.fill_between(x, density, color='#999999', alpha=.2, zorder=1)
            for cp in truth:
                ax.axvline(cp, color='black', ls='--', lw=2, alpha=.9, zorder=10)
            ax.set(xlim=(0, 1), ylim=(0, HISTOGRAM_DENSITY_LIMIT))
            ax.set_xticks(truth, [f'${cp:.2f}$' for cp in truth])
            ax.set_yticks([0, 8, 16, 24] if grouped_covariances else [0, 12, 24])
            ax.tick_params(width=1, length=4, labelsize=14)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_linewidth(1)
            if row == 0:
                ax.set_title(LABELS[method], fontsize=18, pad=8)
            if column == ncols-1:
                ax.text(1.05, .5, rf'$\delta={delta:g}$', transform=ax.transAxes,
                        rotation=90, va='center', ha='left', fontsize=18)
            peak = float(max(hist.max(), density.max()))
            audit.append(dict(covariance=covariance, delta=delta, method=method,
                pooled_locations=len(values), trials=len(fits),
                failed_fits=sum(f['status'] != 'ok' for f in fits),
                empty_estimates=sum(f['status'] == 'ok' and not len(f['cp']) for f in fits),
                bins=12, bin_edges=edges.tolist(),
                kde_bandwidth=float(cv.best_params_['bandwidth']),
                maximum_density=peak, display_limit=HISTOGRAM_DENSITY_LIMIT,
                density_clipped=peak > HISTOGRAM_DENSITY_LIMIT))
    if grouped_covariances:
        for index, covariance in enumerate(COVARIANCES):
            left = axes[0, 3*index].get_position().x0
            right = axes[0, 3*index+2].get_position().x1
            fig.text((left+right)/2, .97, TITLES[covariance], ha='center', va='top', fontsize=18)
    else:
        headings = PAPER_COVARIANCE_TITLES
        for column, (covariance, _) in enumerate(columns):
            box = axes[0, column].get_position()
            fig.text((box.x0+box.x1)/2, .97, headings[covariance],
                     ha='center', va='top', fontsize=18)
    fig.text(.51, .01, 'Fractional location', ha='center', va='bottom', fontsize=18)
    fig.text(.01, .49, 'Density', va='center', rotation='vertical', fontsize=18)
    path = _export(fig, output_dir, stem, artifacts)
    return path, audit


def export_five_column_unknown_histogram(root, output_dir=None, *, expected_truth=(.45, .75)):
    """Export AR(1) WERM+CV/P and all three AMP designs as a 2-by-5 histogram."""
    import matplotlib.pyplot as plt
    from tools.consistent_plotting import CANONICAL_STYLE

    root = Path(root).resolve()
    output_dir = (Path(output_dir).resolve() if output_dir is not None
                  else root / 'figures' / 'compact_unknown')
    spec, groups, input_hashes = _validated_groups(root, expected_truth=expected_truth)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    with plt.rc_context(CANONICAL_STYLE):
        path, histograms = _histogram(groups, spec['unknown_truth'], output_dir, artifacts,
            columns=FIVE_COLUMN_HISTOGRAM, stem='unknown_gap10_T10_hist_five_columns')
    for filename, digest in input_hashes.items():
        if _sha256(root / filename) != digest:
            raise ValueError(f'Saved input changed during plotting: {filename}')
    audit = dict(root=str(root), truth=spec['unknown_truth'],
        columns=[dict(covariance=c, method=m) for c, m in FIVE_COLUMN_HISTOGRAM],
        histogram_deltas=list(HISTOGRAM_DELTAS), histograms=histograms,
        input_sha256=input_hashes, figures=artifacts)
    (output_dir / 'five_column_histogram_sources.json').write_text(json.dumps(audit, indent=2)+'\n')
    return path


def export_paper_unknown_panels(root, *, expected_truth):
    """Export the three count panels and five-column histogram for the appendix."""
    import matplotlib.pyplot as plt
    from tools.consistent_plotting import CANONICAL_STYLE

    root = Path(root).resolve()
    output_dir = root / 'figures' / 'amp_appendix'
    spec, groups, input_hashes = _validated_groups(root, expected_truth=expected_truth)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    with plt.rc_context(CANONICAL_STYLE):
        paths = _metric_plots(_summary(groups), output_dir, artifacts,
                              metrics=('count',), titles=PAPER_COVARIANCE_TITLES)
        histogram, histograms = _histogram(groups, spec['unknown_truth'], output_dir, artifacts,
            columns=FIVE_COLUMN_HISTOGRAM, stem='unknown_gap10_T10_hist_five_columns')
        paths.append(histogram)
    for filename, digest in input_hashes.items():
        if _sha256(root / filename) != digest:
            raise ValueError(f'Saved input changed during plotting: {filename}')
    audit = dict(root=str(root), truth=spec['unknown_truth'],
        covariance_titles=PAPER_COVARIANCE_TITLES, histogram_y_ticks=[0, 12, 24],
        histogram_columns=[dict(covariance=c, method=m) for c, m in FIVE_COLUMN_HISTOGRAM],
        histogram_deltas=list(HISTOGRAM_DELTAS), histograms=histograms,
        input_sha256=input_hashes, figures=artifacts)
    (output_dir / 'figure_sources.json').write_text(json.dumps(audit, indent=2)+'\n')
    return paths


def export_compact_unknown(root=DEFAULT_ROOT, output_dir=None, *, expected_truth=(.25, .50)):
    """Export three Hausdorff panels, three count panels and a 2-by-9 histogram.

    PNG paths are returned in that order, with identity, AR(1), random orthogonal
    order within each metric. Every PNG has a PDF companion. Summary CSV and
    audit JSON retain the success/empty/failure denominators and source hashes.
    ``expected_truth`` explicitly selects the two fractional change-point
    locations required in the saved specification; it does not modify results.
    """
    import matplotlib.pyplot as plt
    from tools.consistent_plotting import CANONICAL_STYLE

    root = Path(root).resolve()
    output_dir = (Path(output_dir).resolve() if output_dir is not None
                  else root / 'figures' / 'compact_unknown')
    spec, groups, input_hashes = _validated_groups(root, expected_truth=expected_truth)
    summary = _summary(groups)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / 'compact_unknown_audit.json'
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        if previous.get('input_sha256') != input_hashes:
            raise ValueError('Saved inputs changed since the previous compact export')
    artifacts = []
    with plt.rc_context(CANONICAL_STYLE):
        paths = _metric_plots(summary, output_dir, artifacts)
        histogram, histogram_audit = _histogram(groups, spec['unknown_truth'], output_dir, artifacts)
        paths.append(histogram)
    for filename, digest in input_hashes.items():
        if _sha256(root / filename) != digest:
            raise ValueError(f'Saved input changed during plotting: {filename}')
    with (output_dir / 'compact_unknown_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    source_root = Path(__file__).resolve().parents[1]
    audit = dict(study='unknown', root=str(root), specification=spec,
        selected_covariances=list(COVARIANCES), selected_methods=list(METHODS),
        selected_deltas=list(DELTAS), histogram_deltas=list(HISTOGRAM_DELTAS),
        covariance_titles=TITLES,
        hausdorff_convention='Mean and actual 25th/75th percentiles of normalized Hausdorff '
            'distance conditional on a successful nonempty estimate; empties are undefined. '
            'Groups with no nonempty estimate have null summaries and no Hausdorff point.',
        count_convention='Mean and actual 25th/75th percentiles over all successful estimates, '
            'including empty estimates as zero; the true count is two.',
        histogram_convention='Successful fits pooled as cp/n; canonical 12 data-range bins and '
            'fivefold Gaussian KDE; common density display range 0..24 covers every peak. Empty fits contribute '
            'no locations. Variable-count fits contribute one observation per estimated boundary.',
        input_sha256=input_hashes,
        plotting_sources_sha256={str(path.relative_to(source_root)): _sha256(path) for path in
            (Path(__file__), source_root / 'tools/werm_amp_compact_plots.py',
             source_root / 'tools/consistent_plotting.py')},
        style=CANONICAL_STYLE, summary=summary, histograms=histogram_audit, figures=artifacts)
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    return paths
