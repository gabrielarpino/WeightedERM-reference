"""Regression checks for the E-L and E-PRIOR notebook specifications."""

import ast
import hashlib
from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _source(name):
    notebook = nbformat.read(ROOT / name, as_version=4)
    code = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    # Further audit/plot cells may be appended after the preserved experiment.
    assert len(code) >= 2
    return code[:2]


def _function(name, source, namespace):
    module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook>', 'exec'), namespace)
    return namespace[name]


def test_notebooks_contain_real_runs_and_embedded_plot_code():
    for filename in ('E_L_misspecified_L.ipynb', 'E_PRIOR_misspecified_prior.ipynb'):
        run, plot = _source(filename)
        assert 'N_TRIALS = 2 if SMOKE_MODE else 100' in run
        assert 'N_JOBS = 2 if SMOKE_MODE else 36' in run
        assert 'Parallel(n_jobs=N_JOBS' in run
        assert 'weighted_erm(' in run
        assert 'with ARTIFACT.open("rb")' in plot
        assert 'plt.show()' in plot
        assert 'subprocess' not in run


def test_joint_prior_marginals_equal_direct_enumeration():
    run, _ = _source('E_PRIOR_misspecified_prior.ipynb')
    marginalize = _function('marginalize_joint_prior', run, {'np': np})
    configs = np.array([[2, 4], [3, 5]])
    phi = marginalize(configs, 6)
    direct = np.zeros((3, 6))
    for first, second in configs:
        direct[0, :first] += .5
        direct[1, first:second] += .5
        direct[2, second:] += .5
    np.testing.assert_allclose(phi, direct)
    np.testing.assert_allclose(phi.sum(axis=0), 1)


def test_unconditional_loss_charges_empty_and_wrong_cardinality():
    run, _ = _source('E_L_misspecified_L.ipynb')
    metrics = _function('error_metrics', run, {'np': np, 'N': 100})
    truth = np.array([20, 50, 80])
    assert metrics([], truth)['unconditional_loss'] == 1.25
    assert np.isnan(metrics([], truth)['conditional_haus'])
    assert metrics(truth, truth)['unconditional_loss'] == 0
    assert metrics([20, 50], truth)['unconditional_loss'] > 0


def test_prior_windows_do_not_overlap_at_shared_boundary():
    run, _ = _source('E_PRIOR_misspecified_prior.ipynb')
    window = _function('in_window', run, {'np': np, 'N': 400})
    indices = np.arange(400)
    assert not np.any(window(indices, .55) & window(indices, .80))


def test_good_prior_has_real_gap_between_configuration_supports():
    run, _ = _source('E_PRIOR_misspecified_prior.ipynb')
    window = _function('in_window', run, {'np': np, 'N': 500})
    indices = np.arange(500)
    first = indices[window(indices, .30, trim_right=.02)]
    second = indices[window(indices, .55, trim_left=.02)]
    assert len(first) and len(second)
    assert first.max() / 500 < .405
    assert second.min() / 500 >= .445
    assert 'GOOD_SUPPORT_GAP = .04' in run
    assert 'first_good = in_window(ALL_CONFIGS[:, 0], 0.30, trim_right=GOOD_SUPPORT_GAP/2)' in run
    assert 'second_good = in_window(ALL_CONFIGS[:, 1], 0.55, trim_left=GOOD_SUPPORT_GAP/2)' in run


def test_exact_conditional_coefficient_mse_matches_noise_monte_carlo():
    run, _ = _source('E_PRIOR_misspecified_prior.ipynb')
    theory = _function('conditional_coefficient_mse', run, {
        'np': np, 'NOISE_SD': .4,
    })
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20, 3))
    beta = rng.normal(size=(3, 2))
    signal = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    phi = np.vstack((np.linspace(1, .1, 20), np.linspace(.1, 1, 20)))
    phi /= phi.sum(axis=0)
    expected = theory(X, beta, signal, phi)
    observed = []
    for _ in range(2000):
        y = np.einsum('ij,ij->i', X, beta[:, signal].T) + rng.normal(0, .4, 20)
        estimate = np.column_stack([
            np.linalg.lstsq(X * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
            for w in phi
        ])
        observed.append(np.mean((estimate-beta)**2))
    assert abs(np.mean(observed) - expected) < .02


def test_pooled_location_and_joint_priors_use_configuration_mass():
    _, plot = _source('E_PRIOR_misspecified_prior.ipynb')
    namespace = {'np': np}
    location = _function('pooled_change_point_prior_mass', plot, namespace)
    joint = _function('exact_joint_prior_mass', plot, namespace)
    configs = np.array([[1, 4], [2, 4], [2, 5]], dtype=int)
    location_mass, x = location(configs, n=6)
    joint_mass = joint(configs, n=6)
    np.testing.assert_allclose(location_mass.sum(), 1)
    np.testing.assert_allclose(joint_mass.sum(), 1)
    np.testing.assert_allclose(
        (joint_mass.sum(axis=1) + joint_mass.sum(axis=0)) / 2,
        location_mass,
    )
    np.testing.assert_allclose(x, np.arange(6) / 6)
    assert location_mass[0] == 0
    assert location_mass[1] == 1 / 6
    assert location_mass[2] == 2 / 6
    assert 'pooled_change_point_prior_mass(configs[name]' in plot
    assert 'exact_joint_prior_mass(configs[name]' in plot
    assert 'np.histogram(' not in plot
    assert 'VISUAL_SUPPORT_GAP' not in plot

    # Uniform over all distinct pairs (Delta=1) gives a flat *pooled* prior.
    all_pairs = np.array([(a, b) for a in range(1, 6) for b in range(a+1, 6)])
    flat_mass, _ = location(all_pairs, n=6)
    np.testing.assert_allclose(flat_mass[1:], flat_mass[1])


def test_prior_notebook_explains_soft_prior_and_discrete_plotting():
    notebook = nbformat.read(ROOT / 'E_PRIOR_misspecified_prior.ipynb', as_version=4)
    prose = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    assert 'not a hard constraint' in prose
    assert '33 of 100' in prose
    assert '12 or 13' in prose
    assert 'exact discrete' in prose
    assert 'uniformly selected change point' in prose
    assert 'flat when' in prose
    assert 'actual support gap' in prose
    assert 'first admissible index' in prose


def test_paper_ready_descriptions_state_design_and_prior_supports():
    el = nbformat.read(ROOT / 'E_L_misspecified_L.ipynb', as_version=4).cells[0].source
    prior = nbformat.read(ROOT / 'E_PRIOR_misspecified_prior.ipynb', as_version=4).cells[0].source
    for prose in (el, prior):
        assert '### Paper-ready experiment description' in prose
        assert '100 paired' in prose
        assert '$n=500$' in prose and '$p=100$' in prose
        assert '0.5^2' in prose and '1.5^2' in prose
        assert 'unpenalized least-squares WERM' in prose
    assert '$L^*=4$' in el
    assert '$\\Delta_n=1$' in el
    assert '$L\\in\\{1,2,3,4,5,6\\}$' in el
    assert 'fixed-$L$' in el
    assert '$\\Delta_n=25$' in prior
    assert '0.175,0.405' in prior and '0.445,0.675' in prior
    assert '0.175,0.425' in prior and '0.675,0.925' in prior
    assert 'all gap-admissible partitions' in prior


def test_final_e_experiment_plot_presentation():
    _, el = _source('E_L_misspecified_L.ipynb')
    _, prior = _source('E_PRIOR_misspecified_prior.ipynb')
    for plot in (el, prior):
        assert 'fig.tight_layout(' in plot
        assert 'MaxNLocator' in plot
        assert 'coefficient_mse' not in plot
        assert 'population_coefficient_bias' not in plot
        assert 'population_loss' not in plot
        assert 'arrays["runtime"]' not in plot
        assert 'arrays["unconditional_loss"]' not in plot
        assert '"No. selected changes"' not in plot
        assert '"Fractional location"' in plot
        assert 'density=True' in plot
        assert "Hausdorff dist./$n$" in plot
        assert 'Conditional Hausdorff / n' not in plot
    assert 'names = ["Bad", "Neutral", "Good"]' in prior
    assert 'ax.set_title(rf"Working $\\hat{{L}}={L}$")' in el
    assert el.count('ax.set_xlabel(r"Working $\\hat{L}$")') == 2
    assert 'set_xlabel("Working L")' not in el
    assert 'axes[0, 0].set_ylabel("Density")' in el
    assert 'axes[1, 0].set_ylabel("Density")' in el
    assert prior.count('ax.set_title(name)') == 4
    assert 'ax.scatter([0], [0]' not in prior
    assert prior.count('axes[0].set_ylabel("Density")') == 2
    assert 'ax.set_ylabel("Prior prob.")' in prior or 'axes[0].set_ylabel("Prior prob.")' in prior
    assert 'axes[0].set_ylabel("Sample weights")' in prior
    assert 'VISUAL_SUPPORT_GAP' not in prior
    assert 'ax.axvspan(' not in prior
    assert 'label="Randomly selected change point"' not in prior
    assert 'color=colors[name], linestyle=line_styles[l]' in prior
    for plot in (el, prior):
        assert '"text.usetex": True' in plot
        assert '"Computer Modern Roman"' in plot


def test_el_uses_cv_selection_within_each_upper_bound():
    notebook = nbformat.read(ROOT / 'E_L_upper_bound.ipynb', as_version=4)
    run, plot = notebook.cells[1].source, notebook.cells[3].source
    assert 'weighted_erm_joint_cv(' in run
    assert 'Lmax=int(upper)' in run
    assert "num_folds=10,n_jobs=1" in run
    assert "delta_fraction=1/20" in run
    assert "absolute_delta_n=1" not in run
    assert "EL_SETTINGS=[(1000,100)]" in run
    assert "from WERM.experiment_utils import weighted_erm_joint_cv" in run
    assert "def el_data(n,p,trial):" in run
    assert "np.abs(np.asarray(prediction)-np.asarray(truth))" in run
    assert "penalty_norm='none'" in run
    assert "cv_scores" in plot
    assert 'Upper bound' not in plot
    assert 'Selected number of signals' in plot
    assert 'Fractional location' in plot
    assert 'manage_ticks=False' in plot
    assert 'FixedLocator(L_values)' in plot


def test_original_el_cells_remain_unchanged():
    notebook = nbformat.read(ROOT / 'E_L_misspecified_L.ipynb', as_version=4)
    expected = (
        'db3ee980d99c9611639e1461d4452f726eca9209f37e4364859baf04ee9b3a38',
        '2d9b1d99353409e8f710f2317321576abdbc4dbdd19fe18d60b5564acb37bc83',
        '51ca261fc6ba69594cae5a51914f139abe2b8d9a78d0c982989d20717adf0c75',
        '586137fd9d82f8d48e8652f24b95e6d6095018109a1b4eaf6c60660bf400fde3',
    )
    assert tuple(hashlib.sha256(cell.source.encode()).hexdigest()
                 for cell in notebook.cells[:4]) == expected


def test_eprior_reports_location_boxplot_mean_and_median():
    notebook = nbformat.read(ROOT / 'E_PRIOR_misspecified_prior.ipynb', as_version=4)
    prose = notebook.cells[0].source
    _, plot = _source('E_PRIOR_misspecified_prior.ipynb')
    assert 'mean / median conditional Hausdorff' in plot
    assert "np.nanmedian(conditional)" in plot
    assert 'Bad: $0.133/0.140$' in prose
    assert 'Neutral: $0.027/0.014$' in prose
    assert 'Good: $0.006/0.004$' in prose
