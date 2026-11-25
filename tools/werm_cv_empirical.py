"""Exact unpenalized squared-loss WERM-CV with auditable intermediate fits.

This is the exhaustive production CV protocol, specialized to squared loss:
interlaced folds, fixed-count WERM training partitions, unweighted segment
refits, absolute held-out prediction errors, and a full-data WERM refit.
Only the additive-loss partition dynamic program is accelerated. Neither
fitting nor count selection receives the true coefficients or change points.
"""
import numpy as np

from WERM.proximal import fit_weighted
from tools.werm_cv_partition import additive_partition


def minimum_gap(n, delta_fraction):
    """Fold-local spacing used by weighted_erm_joint_cv."""
    return max(int(np.ceil(n / 50.)), int(float(delta_fraction) * n), 1)


def get_weights(weights, n, count, gap):
    key = (int(n), int(count), int(gap))
    phi = np.asarray(weights[key] if key in weights else weights[n, count], dtype=float)
    if phi.shape != (count, n) or not np.isfinite(phi).all() or np.any(phi < 0):
        raise ValueError(f'Invalid marginal weights for {key}')
    if not np.allclose(phi.sum(axis=0), 1., atol=1e-8, rtol=1e-8):
        raise ValueError(f'Marginal weights must have unit column sums: {key}')
    if np.any(phi.sum(axis=1) <= 0):
        raise ValueError(f'Marginal weights must have positive row sums: {key}')
    return phi


def least_squares(X, y, weights, loss_scale=1.):
    result = fit_weighted(X, y, weights, loss='least_squares',
                          loss_scale=loss_scale, penalty='none', penalty_strength=0.)
    if not result.converged:
        raise RuntimeError('Unpenalized least-squares fit failed KKT tolerance')
    return result


def fit_partition(X, y, count, weights, delta_fraction, loss_scale=1.,
                  return_coefficients=False, minimum_segment_fraction=None):
    n = len(y)
    gap = minimum_gap(n, delta_fraction)
    end_min = 1
    if minimum_segment_fraction is not None:
        if not 0 < minimum_segment_fraction <= 1:
            raise ValueError('minimum_segment_fraction must lie in (0,1]')
        end_min = max(1,int(np.ceil(minimum_segment_fraction*n)))
        gap = max(gap,end_min)
    phi = get_weights(weights, n, count, gap)
    fits = [least_squares(X, y, w/w.sum(), loss_scale) for w in phi]
    coefficients = np.column_stack([fit.coef for fit in fits])
    cp = additive_partition((X @ coefficients-y[:,None])**2,gap,end_min)
    output = dict(cp=cp, segments=count, n=n, gap=gap, end_min=end_min,
                  max_kkt=max(fit.kkt_violation for fit in fits))
    if return_coefficients:
        output['coefficients'] = coefficients
    return output


def refit_and_score(X, y, train, test, training_cp, loss_scale=1.,
                    return_coefficients=False):
    """Map boundaries, hard-refit with M, and score only with absolute loss."""
    n, p = X.shape
    training_cp = np.asarray(training_cp, dtype=int)
    mapped = train[training_cp[(training_cp > 0) & (training_cp < len(train))]]
    boundaries = np.r_[0, mapped, n]
    coefficients, sizes, residuals = [], [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        rows = train[(train >= start) & (train < stop)]
        sizes.append(len(rows))
        if len(rows):
            fitted = least_squares(X[rows], y[rows], np.ones(len(rows)), loss_scale)
            coefficients.append(fitted.coef)
            residuals.append(fitted.kkt_violation)
        else:
            coefficients.append(np.zeros(p))
            residuals.append(0.)
    coefficients = np.asarray(coefficients)
    segments = np.searchsorted(boundaries[1:], test, side='right')
    segments = np.minimum(segments, len(coefficients)-1)
    predictions = np.einsum('ij,ij->i', X[test], coefficients[segments])
    absolute_errors = np.abs(predictions-y[test])
    output = dict(mapped_cp=mapped, boundaries=boundaries,
                  segment_sizes=np.asarray(sizes, dtype=int),
                  segment_sample_ratios=np.asarray(sizes)/p,
                  coefficient_norms=np.linalg.norm(coefficients, axis=1),
                  validation_sum=float(absolute_errors.sum()),
                  validation_mean=float(absolute_errors.mean()),
                  max_refit_kkt=float(max(residuals)))
    if return_coefficients:
        output.update(segment_coefficients=coefficients, predictions=predictions,
                      absolute_errors=absolute_errors)
    return output


def run_cv(X, y, max_segments, weights, delta_fraction, folds=10,
           loss_scale=1., return_coefficients=False, minimum_segment_fraction=None):
    """Compute all candidate scores once; return selection for every upper bound.

    ``weights`` maps ``(sample_size, segment_count, gap)`` to frozen marginal
    matrices; ``(sample_size, segment_count)`` keys are also accepted when the
    caller guarantees a single gap convention. The raw ``cv_scores`` follow
    the production helper's sum convention. Divide by n (also provided as
    ``normalized_cv_scores``) for a normalized CV risk. The optional
    ``minimum_segment_fraction`` constrains all segments, including endpoints,
    to at least ceil(fraction * fit_sample_size) rows. It is a deliberate
    variation from the original feasible set; caller-supplied weights must
    match that variation.
    """
    X, y = np.asarray(X), np.asarray(y)
    n, p = X.shape
    if y.shape != (n,) or not (2 <= folds <= n) or max_segments < 1:
        raise ValueError('Invalid CV dimensions or candidate count')
    if not 0 < delta_fraction < 1 or loss_scale <= 0:
        raise ValueError('Invalid spacing or loss scale')
    indices = np.arange(n)
    fold_scores = np.empty((max_segments, folds))
    fold_results = {}
    for count in range(1, max_segments+1):
        for fold in range(folds):
            test = indices[fold::folds]
            train = indices[indices % folds != fold]
            fitted = fit_partition(X[train], y[train], count, weights,
                                   delta_fraction, loss_scale,
                                   minimum_segment_fraction=minimum_segment_fraction)
            score = refit_and_score(X, y, train, test, fitted['cp'], loss_scale,
                                    return_coefficients)
            score.update(train_cp=fitted['cp'], training_n=len(train),
                         validation_n=len(test), gap=fitted['gap'],end_min=fitted['end_min'],
                         max_partition_kkt=fitted['max_kkt'])
            fold_scores[count-1, fold] = score['validation_sum']
            fold_results[count, fold] = score
    totals = fold_scores.sum(axis=1)
    if not np.isfinite(totals).all():
        raise RuntimeError('Nonfinite CV scores')
    full_fits = {count: fit_partition(X, y, count, weights, delta_fraction,
                                     loss_scale, return_coefficients,
                                     minimum_segment_fraction)
                 for count in range(1, max_segments+1)}
    selected = {}
    for bound in range(1, max_segments+1):
        winner = int(np.argmin(totals[:bound]))+1
        selected[bound] = dict(selected_segments=winner,
                               cp=full_fits[winner]['cp'].copy(),
                               validation_mean=float(totals[winner-1]/n))
    return dict(n=n, p=p, folds=folds, max_segments=max_segments,
                delta_fraction=delta_fraction, loss_scale=loss_scale,
                minimum_segment_fraction=minimum_segment_fraction,
                fold_scores=fold_scores, cv_scores=totals,
                normalized_cv_scores=totals/n, fold_results=fold_results,
                full_fits=full_fits, selected_by_bound=selected)


def gaussian_validation_risk(result, beta, true_labels, noise_variance):
    """Post-fit oracle diagnostic; never used to fit or select empirical models.

    Conditional on a fold's training data and fixed true coefficients, its
    independent Gaussian held-out residual has variance sigma^2 + ||error||²/n
    for the isotropic design X_ij ~ N(0,1/n). Requires stored segment coefficients.
    """
    beta, true_labels = np.asarray(beta), np.asarray(true_labels)
    n = result['n']
    risks = np.zeros_like(result['fold_scores'])
    for (count, fold), fit in result['fold_results'].items():
        test = np.arange(fold, n, result['folds'])
        assigned = np.searchsorted(fit['boundaries'][1:], test, side='right')
        errors = fit['segment_coefficients'][assigned] - beta[:,true_labels[test]].T
        variance = noise_variance + np.sum(errors**2,axis=1)/n
        risks[count-1,fold] = np.sqrt(2/np.pi)*np.sqrt(variance).sum()
    return risks.sum(axis=1)/n
