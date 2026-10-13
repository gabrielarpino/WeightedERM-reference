import numpy as np
from tools.myocardial_final_final_gram import perturb_rho

def test_sensitivity_preserves_baseline_and_scales_eigenvalues():
    rho=np.array([[2.,.3],[.3,1.]])
    np.testing.assert_array_equal(perturb_rho(rho,1,1),rho)
    expected=np.linalg.eigvalsh(rho)*[.8,1.2]
    np.testing.assert_allclose(np.linalg.eigvalsh(perturb_rho(rho,.8,1.2)),expected)

def test_builder_contains_three_fixed_candidates_and_reference_format():
    from tools.build_myocardial_final_final import build,NAME
    import nbformat
    build();n=nbformat.read(NAME,4)
    code='\n'.join(c.source for c in n.cells if c.cell_type=='code')
    assert "('config283','p')" in code and "('config189','n')" in code
    assert 'figsize=(12, 2.0)' in code and 'figsize=(13*5/3, 7*5/3)' in code
    assert "'text.usetex': True" in code
