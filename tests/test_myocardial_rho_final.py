"""Scientific-contract tests for the checkpoint-anchored notebook."""
import ast
from pathlib import Path
import nbformat
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'Myocardial Infarction Experiment - rho sensitivity final.ipynb'

def notebook():
    return nbformat.read(PATH, 4)

def function(name):
    for cell in notebook().cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(cell.source, node)
    raise AssertionError(name)

def test_original_figure_function_unchanged():
    original = nbformat.read(ROOT / 'Myocardial Infarction Experiment - rho sensitivity.ipynb', 4)
    tree = ast.parse(original.cells[21].source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    assert function('plot_posterior_compact') == ast.get_source_segment(original.cells[21].source, node)

def test_null_states_are_both_boundaries_and_scaling_invariant():
    ns = {'np': np}
    exec(function('summarize_weights'), ns)
    summarize = ns['summarize_weights']
    a = summarize([1, 6, 3], [[0], [5], [10]], 10)
    b = summarize([7, 42, 21], [[0], [5], [10]], 10)
    assert a['absence'] == pytest.approx(.4)
    assert a['presence'] == pytest.approx(.6)
    np.testing.assert_allclose(a['probabilities'], b['probabilities'])
    assert a['boundary_zero'] == pytest.approx(.1)
    assert a['boundary_n'] == pytest.approx(.3)

@pytest.mark.parametrize('weights', [[0,0,0], [1,-1,1], [1,np.nan,1]])
def test_invalid_weights_rejected(weights):
    ns = {'np': np}
    exec(function('summarize_weights'), ns)
    with pytest.raises(ValueError):
        ns['summarize_weights'](weights, [[0],[5],[10]], 10)

def test_no_redefined_weights_or_fit_or_extra_null():
    source = '\n'.join(c.source for c in notebook().cells if c.cell_type == 'code')
    assert 'weighted_erm(' not in source
    assert 'estimate_rho' not in source
    assert "phi = reference['ϕ']" in source
    assert "eta = np.asarray(reference['η_arr_subsampled']" in source
    assert 'FRESH BASELINE FIXED-POINT REPRODUCTION PASSED' in source
    assert "jax.config.update('jax_enable_x64', True)" in source
    for c in notebook().cells:
        if c.cell_type == 'code': ast.parse(c.source)

def test_boundary_configurations_really_have_no_interior_change():
    from WERM.posterior import η_to_ψ_jax_new
    np.testing.assert_array_equal(η_to_ψ_jax_new(np.array([0]), 8), np.ones(8))
    np.testing.assert_array_equal(η_to_ψ_jax_new(np.array([8]), 8), np.zeros(8))
    np.testing.assert_array_equal(η_to_ψ_jax_new(np.array([-1]), 8), np.zeros(8))
    np.testing.assert_array_equal(η_to_ψ_jax_new(np.array([3]), 8), [0,0,0,1,1,1,1,1])

def test_actual_reference_reconstructs_and_accounts_for_hidden_mass():
    # Import before unpickling so JAX does not silently narrow the archived arrays.
    import WERM.posterior as posterior_module
    import pickle
    from scipy.special import logsumexp
    with (ROOT / 'results/myocardial_infarction_posterior_exact_one_chgpt.pkl').open('rb') as handle:
        reference = pickle.load(handle)
    ns = dict(np=np, posterior_module=posterior_module, logsumexp=logsumexp,
              eta=reference['η_arr_subsampled'], theta=reference['Theta_deb'],
              y=reference['y_prep'], prior=reference['p_η_arr_subsampled'], n=1692)
    exec(function('summarize_weights'), ns)
    exec(function('evaluate'), ns)
    result = ns['evaluate'](reference['ρ_hat'], reference['nu_arr'], reference['kappa_arr'])
    np.testing.assert_allclose(result['weights'], reference['posterior'], rtol=1e-10, atol=1e-12)
    assert result['presence'] == pytest.approx(.93887788, abs=1e-8)
    assert result['absence'] == pytest.approx(.06112212, abs=1e-8)
    assert result['boundary_n'] > 0 and np.isnan(reference['unique_ages'][-1])
    assert np.sum(result['probabilities'][:-1]) < 1

def test_count_prior_is_half_half_and_reweighting_preserves_conditional_locations():
    ns = {'np': np}
    exec(function('summarize_weights'), ns)
    exec(function('apply_count_prior'), ns)
    eta = np.array([[0], [2], [5], [8], [10]])
    flat = ns['apply_count_prior'](dict(log_likelihood=np.zeros(5), weights=np.ones(5)/5), eta, 10)
    assert flat['presence'] == pytest.approx(.5)
    assert flat['absence'] == pytest.approx(.5)
    np.testing.assert_allclose(flat['count_prior'], [.25, 1/6, 1/6, 1/6, .25])
    weights = np.array([1., 2., 4., 3., 2.])
    changed = ns['apply_count_prior'](dict(log_likelihood=np.log(weights), weights=weights), eta, 10)
    np.testing.assert_array_equal(changed['reference_weights'], weights)
    np.testing.assert_allclose(changed['probabilities'][1:-1]/changed['presence'], weights[1:-1]/weights[1:-1].sum())
