"""Paired, read-only summaries and figures for AMP at T=10 and T=20."""
from pathlib import Path

import numpy as np
import pandas as pd


COVARIANCE_TITLES = {'identity': r'$\Sigma=I$',
    'ar1': r'AR(1), $r=0.4$', 'm3': 'M3 random correlation'}
STYLES = {10: ('#EE7733', 'o', '-.'), 20: ('#4477AA', 's', '-')}


def _paired_records(t10_payload, t20_payload):
    """Pair every completed T20 trial; permit a partial T20 checkpoint set."""
    for key in ('p', 'deltas', 'covariances', 'studies', 'unknown_truth',
                'known_truth', 'separation_fraction', 'noise_std', 'max_segments'):
        if t10_payload['specification'][key] != t20_payload['specification'][key]:
            raise ValueError(f'Iteration comparison has mismatched {key}')
    def index(payload):
        result = {}
        for record in payload['records']:
            key = (record['study'], record['covariance'], record['delta'], record['trial'])
            if key in result:
                raise ValueError(f'Duplicate trial: {key}')
            result[key] = record
        return result
    old, new = index(t10_payload), index(t20_payload)
    if not new.keys() <= old.keys():
        raise ValueError('A completed T20 trial has no matching T10 record')
    pairs = {}
    for key, current in new.items():
        previous = old[key]
        if previous['data_hash'] != current['data_hash']:
            raise ValueError(f'Unpaired data: {key}')
        if not np.array_equal(previous['truth'], current['truth']):
            raise ValueError(f'Mismatched ground truth: {key}')
        a, b = previous['methods']['AMP'], current['methods']['AMP']
        # A failed adapter can lack a seed; it remains an explicitly failed
        # fit on the same dataset rather than being silently discarded.
        if a.get('seed') is not None and b.get('seed') is not None and a['seed'] != b['seed']:
            raise ValueError(f'Mismatched AMP seed: {key}')
        pairs[key] = (previous, current)
    return pairs


def _valid_values(fits, field):
    values = np.asarray([f.get(field, np.nan) for f in fits
                         if f.get('status') == 'ok'], dtype=float)
    return values[np.isfinite(values)]


def paired_summary(t10_payload, t20_payload):
    """Return conditional metrics and common-nonempty paired differences.

    Failure, empty and correct-count rates use all paired datasets as their
    denominator.  Hausdorff means condition on successful nonempty estimates;
    their paired difference instead uses the common nonempty subset.  A
    separate bounded diagnostic assigns loss one to empty or failed fits.
    """
    pairs = _paired_records(t10_payload, t20_payload)
    groups = {}
    for key, pair in pairs.items():
        groups.setdefault(key[:3], []).append(pair)
    rows = []
    for (study, covariance, delta), group in sorted(groups.items()):
        row = dict(study=study, covariance=covariance, delta=delta, trials=len(group))
        truth_count = len(group[0][0]['truth'])
        for index, iterations in enumerate((10, 20)):
            fits = [pair[index]['methods']['AMP'] for pair in group]
            successes = np.array([f.get('status') == 'ok' for f in fits])
            empties = np.array([ok and f.get('count') == 0 for ok, f in zip(successes, fits)])
            haus = _valid_values(fits, 'haus')
            counts = _valid_values(fits, 'count')
            bounded = [float(f['haus']) if ok and np.isfinite(f.get('haus', np.nan)) else 1.
                       for ok, f in zip(successes, fits)]
            suffix = f'_T{iterations}'
            row.update({
                'successes'+suffix: int(successes.sum()),
                'nonempty_successes'+suffix: len(haus),
                'failure_rate'+suffix: float(1-successes.mean()),
                'empty_rate'+suffix: float(empties.mean()),
                'mean_haus'+suffix: float(haus.mean()) if len(haus) else np.nan,
                'mean_count'+suffix: float(counts.mean()) if len(counts) else np.nan,
                'correct_count_rate'+suffix: float(np.mean([
                    ok and f.get('count') == truth_count for ok, f in zip(successes, fits)])),
                'mean_haus_empty_or_failure_one'+suffix: float(np.mean(bounded)),
            })
        common_haus, count_differences = [], []
        changed_locations = []
        for previous, current in group:
            a, b = previous['methods']['AMP'], current['methods']['AMP']
            if a.get('status') == b.get('status') == 'ok':
                count_differences.append(b['count']-a['count'])
                changed_locations.append(not np.array_equal(a['cp'], b['cp']))
                if np.isfinite(a.get('haus', np.nan)) and np.isfinite(b.get('haus', np.nan)):
                    common_haus.append(b['haus']-a['haus'])
        row['n_common_nonempty'] = len(common_haus)
        row['mean_paired_haus_T20_minus_T10'] = float(np.mean(common_haus)) if common_haus else np.nan
        row['n_common_successful'] = len(count_differences)
        row['mean_paired_count_T20_minus_T10'] = float(np.mean(count_differences)) if count_differences else np.nan
        row['changed_locations_rate_common_successful'] = float(np.mean(changed_locations)) if changed_locations else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_amp_iteration_comparison(t10_payload, t20_payload, study, output_dir):
    """One 2x3 Hausdorff/count figure, with separate covariance columns."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    cfg = t20_payload['specification']
    pairs = {key: pair for key, pair in _paired_records(t10_payload, t20_payload).items()
             if key[0] == study}
    if not pairs:
        return []
    covariances, deltas = cfg['covariances'], cfg['deltas']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style = {'font.family':'serif', 'font.size':11, 'axes.labelsize':13,
             'legend.fontsize':10, 'axes.spines.top':False, 'axes.spines.right':False,
             'axes.linewidth':1.1, 'text.usetex':True, 'savefig.dpi':180}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, len(covariances), figsize=(3.65*len(covariances),5.7),
                                 sharex=True, sharey='row', squeeze=False)
        haus_limits = []
        for col, covariance in enumerate(covariances):
            groups = {delta: [pair for key,pair in pairs.items()
                             if key[1] == covariance and key[2] == delta] for delta in deltas}
            for row, field in enumerate(('haus', 'count')):
                ax = axes[row, col]
                for index, iterations in enumerate((10,20)):
                    color, marker, line = STYLES[iterations]
                    means, q25, q75 = [], [], []
                    for delta in deltas:
                        fits = [pair[index]['methods']['AMP'] for pair in groups[delta]]
                        values = _valid_values(fits, field)
                        if len(values):
                            means.append(values.mean())
                            low, high = np.quantile(values,[.25,.75])
                            q25.append(low); q75.append(high)
                            if field == 'haus':
                                haus_limits.extend((float(values.mean()), float(high)))
                        else:
                            # Preserve a gap in the curve when no fit provides
                            # this metric; do not connect across missing ratios.
                            means.append(np.nan); q25.append(np.nan); q75.append(np.nan)
                    ax.plot(deltas,means,label=rf'AMP, $T={iterations}$',color=color,
                            marker=marker,ls=line,lw=2,ms=6,markeredgecolor='white',markeredgewidth=.8)
                    ax.vlines(deltas,q25,q75,color=color,alpha=.6,lw=1.3)
                    for x,low,high in zip(deltas,q25,q75):
                        if np.isfinite(low) and np.isfinite(high):
                            ax.hlines([low,high],x-.07,x+.07,color=color,lw=1.3)
                ax.set_xticks(deltas)
                ax.yaxis.set_major_locator(MaxNLocator(5,integer=field=='count'))
                if row == 1:
                    ax.set_xlabel(r'$\delta=n/p$')
                    ax.axhline(len(cfg[study+'_truth']),color='gray',ls=':',lw=1)
                    ax.set_ylim(-.15,cfg['max_segments']-.65)
                else:
                    ax.set_title(COVARIANCE_TITLES[covariance])
        # Set the shared limit only after all covariance panels are drawn;
        # fixing it at the first panel would clip later panels' quartiles.
        axes[0,0].set_ylim(0, max(1e-3, max(haus_limits))*1.08 if haus_limits else 1.)
        axes[0,0].set_ylabel('Hausdorff error')
        axes[1,0].set_ylabel('No. change points')
        axes[1,-1].legend(loc='lower right',frameon=True,framealpha=.9)
        gap = 1/cfg['separation_fraction']
        scenario = 'Unknown count' if study=='unknown' else r'Known count, $L=4$'
        fig.suptitle(scenario+rf'; $\Delta_n=n/{gap:g}$, $p={cfg["p"]}$')
        diagnostics = []
        for index,iterations in enumerate((10,20)):
            fits = [pair[index]['methods']['AMP'] for pair in pairs.values()]
            failed = sum(f.get('status') != 'ok' for f in fits)
            empty = sum(f.get('status') == 'ok' and f.get('count') == 0 for f in fits)
            diagnostics.append(f'$T={iterations}$: {empty} empty, {failed} failed')
        fig.text(.5,.015,f'{len(pairs)} paired datasets; '+ '; '.join(diagnostics)+'.\n'
                 'Means and quartiles. Hausdorff error conditions on successful nonempty estimates.',
                 ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.075,1,.96),w_pad=1.1,h_pad=1.0)
        stem = f'{study}_AMP_T10_vs_T20_gap{gap:g}'
        for ext in ('pdf','png'):
            fig.savefig(output_dir/f'{stem}.{ext}',bbox_inches='tight')
        plt.close(fig)
    return [output_dir/f'{stem}.png']
