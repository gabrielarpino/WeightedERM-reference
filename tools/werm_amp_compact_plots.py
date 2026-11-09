"""Canonical, compact presentation of the saved known-count WERM/AMP study.

This module only reads completed checkpoints.  It neither fits an estimator nor
changes the experiment.  Styles, histogram bins and KDE follow the approved
M1/M2/M3 final notebooks via ``tools.consistent_plotting.CANONICAL_STYLE``.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from tools.werm_amp_experiment import (
    COVARIANCE_TITLES, METHOD_STYLES, load_results,
)


DEFAULT_ROOT = (Path(__file__).resolve().parents[1]
                / 'results/WERM_AMP_comparison/e8c4b4c6af5f')
COVARIANCES = ('identity', 'ar1', 'm3')
METHODS = ('WERM', 'AMP')
DELTAS = (1.5, 3.5, 7.5, 11.5)
HISTOGRAM_DELTAS = (7.5, 11.5)
LABELS = {'WERM': r'\textbf{WERM}', 'AMP': 'AMP'}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validated_groups(root):
    """Require all 100 paired known-count trials at every plotted setting."""
    payload = load_results(root)
    spec = payload['specification']
    required = {'trials': 100, 'max_segments': 4, 'amp_iterations': 10,
                'whitening': False, 'response_standardization': False,
                'coefficient_loss': 'squared', 'coefficient_penalty': 'none'}
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise ValueError(f'Expected {key}={expected!r}; found {spec.get(key)!r}')
    if (tuple(spec['deltas']) != DELTAS
            or set(spec['covariances']) != set(COVARIANCES)
            or not np.isclose(spec['separation_fraction'], .1)
            or not np.allclose(spec['known_truth'], [.25, .45, .70])
            or not np.isclose(spec['ar1_correlation'], .4)):
        raise ValueError('Expected the saved n/10, known-L=4 covariance comparison')
    groups = {(covariance, delta): []
              for covariance in COVARIANCES for delta in DELTAS}
    seen = set()
    hashes = {'specification.json': _sha256(root / 'specification.json')}
    for record in payload['records']:
        if record['study'] != 'known':
            continue
        key = (record['covariance'], record['delta'])
        trial_key = (*key, record['trial'])
        if key not in groups or trial_key in seen:
            raise ValueError(f'Unexpected or duplicate known-count trial: {trial_key}')
        seen.add(trial_key)
        if record['spec_digest'] != root.name:
            raise ValueError(f'Checkpoint belongs to a different run: {trial_key}')
        n = round(record['delta'] * spec['p'])
        truth = np.rint(n * np.asarray(spec['known_truth'])).astype(int)
        if (record['n'] != n or record['p'] != spec['p']
                or not np.array_equal(record['truth'], truth)):
            raise ValueError(f'Inconsistent checkpoint dimensions/truth: {trial_key}')
        if set(record['methods']) != set(METHODS):
            raise ValueError(f'Expected exactly WERM and AMP: {trial_key}')
        for method, fit in record['methods'].items():
            if fit.get('status') != 'ok':
                continue  # Report failed fits in the audit; never impute locations.
            cp = np.asarray(fit['cp'])
            if (len(cp) != 3 or fit['count'] != 3
                    or not np.all((cp > 0) & (cp < n))
                    or not np.all(np.diff(cp) > 0)
                    or not np.array_equal(cp, cp.astype(int))):
                raise ValueError(f'Invalid fixed-count output: {trial_key}, {method}')
            distances = np.abs(cp[:, None] - truth[None, :]) / n
            haus = max(distances.min(axis=0).max(), distances.min(axis=1).max())
            if not np.isclose(fit['haus'], haus, rtol=1e-12, atol=1e-12):
                raise ValueError(f'Incorrect normalized Hausdorff metric: {trial_key}, {method}')
            if method == 'AMP' and (fit.get('iterations_requested') != 10
                                    or not fit.get('fixed_count')
                                    or fit.get('design_transform') != 'none'):
                raise ValueError(f'Unexpected AMP settings: {trial_key}')
        groups[key].append(record)
        filename = (f'known__{record["covariance"]}__d{record["delta"]:g}'
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
    rows = []
    for covariance in COVARIANCES:
        for delta in DELTAS:
            for method in METHODS:
                fits = [r['methods'][method] for r in groups[covariance, delta]]
                values = np.asarray([f['haus'] for f in fits if f['status'] == 'ok'])
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError(f'No finite Hausdorff summary: {covariance}, {delta}, {method}')
                q25, median, q75 = np.quantile(values, [.25, .5, .75])
                rows.append(dict(covariance=covariance, delta=delta, method=method,
                    trials=len(fits), successes=len(values), failures=len(fits)-len(values),
                    mean=float(values.mean()), q25=float(q25), median=float(median),
                    q75=float(q75), mean_count=3.,
                    amp_iterations_completed_min=(min(f['iterations_completed'] for f in fits
                        if f['status'] == 'ok') if method == 'AMP' else None),
                    amp_iterations_completed_max=(max(f['iterations_completed'] for f in fits
                        if f['status'] == 'ok') if method == 'AMP' else None)))
    return rows


def _export(fig, output_dir, name, artifacts):
    """Use canonical tight bounds; return the raster alongside a vector PDF."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1)
    paths = {}
    for extension in ('pdf', 'png'):
        path = output_dir / f'{name}.{extension}'
        fig.savefig(path, bbox_inches=bounds, dpi=300)
        paths[extension] = dict(path=path.name, sha256=_sha256(path))
    artifacts.append(dict(name=name, canvas_inches=fig.get_size_inches().tolist(),
        exported_inches=[bounds.width, bounds.height], files=paths,
        axes=[dict(xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), title=ax.get_title(),
                   xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim())) for ax in fig.axes],
        legends=sum(ax.get_legend() is not None for ax in fig.axes)))
    plt.close(fig)
    return output_dir / f'{name}.png'


def _hausdorff_plots(summary, output_dir, artifacts):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    upper = 1.08 * max(max(row['mean'], row['q75']) for row in summary)
    paths = []
    for index, covariance in enumerate(COVARIANCES):
        fig, ax = plt.subplots(figsize=(5., 4.))
        for method in METHODS:
            rows = [row for row in summary
                    if row['covariance'] == covariance and row['method'] == method]
            x = [row['delta'] for row in rows]
            q25, q75 = [row['q25'] for row in rows], [row['q75'] for row in rows]
            color, marker, line = METHOD_STYLES[method]
            # Separate mean markers from the actual quartiles, since a skewed
            # sample can have its mean outside the interquartile interval.
            ax.vlines(x, q25, q75, color=color, linewidth=3.5, alpha=.8)
            for quantile in (q25, q75):
                ax.plot(x, quantile, ls='none', marker='_', markersize=8,
                        markeredgewidth=2, color=color, alpha=.8)
            ax.plot(x, [row['mean'] for row in rows], label=LABELS[method],
                    color=color, marker=marker, ls=line, linewidth=3.5,
                    markersize=8, markeredgewidth=2, markeredgecolor='white', alpha=.8)
        # Leave room below zero for the near-zero AMP markers at large delta.
        ax.set(xlabel=r'$\delta=n/p$', ylabel=r'Hausdorff dist. / $n$',
               title=COVARIANCE_TITLES[covariance],
               ylim=(-.03 * upper, max(.001, upper)))
        ax.set_xticks(DELTAS, [f'{value:g}' for value in DELTAS])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis='both', which='major', width=1.5, length=6, labelsize=14)
        if index == 0:
            ax.legend(loc='best', fancybox=True, framealpha=.4)
        fig.tight_layout()
        paths.append(_export(fig, output_dir, f'known_gap10_T10_{covariance}_haus', artifacts))
    return paths


def _histogram(groups, truth, output_dir, artifacts):
    """The canonical histogram algorithm, with covariance headings in pairs.

    The final-notebook helper exports/displays immediately, so its short plotting
    loop is reproduced here to add group headings without changing global state.
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    fig, axes = plt.subplots(2, 6, figsize=(15., 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.965, bottom=.18, top=.80,
                        wspace=.05, hspace=.08)
    audit = []
    for row, delta in enumerate(HISTOGRAM_DELTAS):
        for column, (covariance, method) in enumerate(
                (c, m) for c in COVARIANCES for m in METHODS):
            ax = axes[row, column]
            records = groups[covariance, delta]
            values = np.concatenate([np.asarray(record['methods'][method]['cp']) / record['n']
                for record in records if record['methods'][method]['status'] == 'ok'])
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
            ax.set(xlim=(0, 1), ylim=(0, 18))
            ax.set_xticks(truth, [f'${cp:.2f}$' for cp in truth])
            ax.set_yticks([0, 8, 16])
            ax.tick_params(width=1, length=4, labelsize=14)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_linewidth(1)
            if row == 0:
                ax.set_title(LABELS[method], fontsize=18, pad=8)
            if column == 5:
                ax.text(1.05, .5, rf'$\delta={delta:g}$', transform=ax.transAxes,
                        rotation=90, va='center', ha='left', fontsize=18)
            audit.append(dict(covariance=covariance, delta=delta, method=method,
                pooled_locations=len(values), bins=12, bin_edges=edges.tolist(),
                kde_bandwidth=float(cv.best_params_['bandwidth']),
                maximum_density=float(max(hist.max(), density.max())), display_limit=18.))
    for index, covariance in enumerate(COVARIANCES):
        left = axes[0, 2*index].get_position().x0
        right = axes[0, 2*index+1].get_position().x1
        title = 'random orthogonal' if covariance == 'm3' else COVARIANCE_TITLES[covariance]
        fig.text((left+right)/2, .97, title,
                 ha='center', va='top', fontsize=18)
    fig.text(.51, .01, 'Fractional location', ha='center', va='bottom', fontsize=18)
    fig.text(.015, .49, 'Density', va='center', rotation='vertical', fontsize=18)
    path = _export(fig, output_dir, 'known_gap10_T10_hist', artifacts)
    return path, audit


def export_compact_known(root=DEFAULT_ROOT, output_dir=None):
    """Export three Hausdorff panels and one two-row/six-column histogram.

    Returns PNG paths in identity, AR(1), M3, histogram order.  Every PNG has a
    matching PDF.  ``compact_known_audit.json`` records settings, input hashes,
    exclusions, exact plotted summaries, histogram diagnostics and file hashes;
    ``compact_known_summary.csv`` provides the numerical curve summaries.
    """
    import matplotlib.pyplot as plt
    from tools.consistent_plotting import CANONICAL_STYLE

    root = Path(root).resolve()
    output_dir = (Path(output_dir).resolve() if output_dir is not None
                  else root / 'figures' / 'compact_known')
    spec, groups, input_hashes = _validated_groups(root)
    summary = _summary(groups)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    with plt.rc_context(CANONICAL_STYLE):
        paths = _hausdorff_plots(summary, output_dir, artifacts)
        histogram, histogram_audit = _histogram(groups, spec['known_truth'], output_dir, artifacts)
        paths.append(histogram)
    with (output_dir / 'compact_known_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    source_root = Path(__file__).resolve().parents[1]
    audit = dict(study='known', root=str(root), specification=spec,
        selected_covariances=list(COVARIANCES), selected_deltas=list(DELTAS),
        histogram_deltas=list(HISTOGRAM_DELTAS),
        summary_convention='Mean and actual 25th/75th percentiles of normalized Hausdorff distance; '
            'failed fits excluded and counted explicitly in each summary row.',
        histogram_convention='Successful fits pooled as cp/n; canonical 12 data-range bins and '
            'fivefold Gaussian KDE; fixed density display range 0..18.',
        input_sha256=input_hashes,
        plotting_sources_sha256={str(path.relative_to(source_root)): _sha256(path) for path in
            (Path(__file__), source_root / 'tools/consistent_plotting.py')},
        style=CANONICAL_STYLE, summary=summary, histograms=histogram_audit, figures=artifacts)
    (output_dir / 'compact_known_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return paths
