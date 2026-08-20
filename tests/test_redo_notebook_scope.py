import hashlib
import json
from pathlib import Path


LEGACY_NOTEBOOK_HASHES = {
    "M1 Comparison Experiments.ipynb": "3cb2d190201b114af68d12e61f2054c20fe4b3d9543c86ce3c4df87304778f58",
    "M2 Comparison Experiments.ipynb": "507dbc43903ed0292def4d804f868ad7731f837f183524365d375fb74dfb34b9",
    "M3 Comparison Experiments.ipynb": "ab6fb3542bc7592f252cf18fe06305451f45fd17a76d936d65c810c761eb4312",
    "M1_upsample.ipynb": "2292f5a0f0c535eeafcda325fe5571889a678de648c820506a545bdf34fb2fa6",
    "M2_upsample.ipynb": "3a41ba08eb256eb7181e4dc1309a2c4cdf8332c9422a4041e43810fdf5705c6b",
    "M3_upsample.ipynb": "c8048dc5895d352eeb9e2ceb6e509dcbab6c5b6586b55d38e27445793c71675e",
}


def _code(path):
    notebook = json.loads(Path(path).read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_legacy_notebooks_are_byte_identical():
    for path, expected in LEGACY_NOTEBOOK_HASHES.items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert actual == expected, path


def test_redo_notebooks_do_not_subtract_one_from_competitor_indices():
    code = _code("M1_upsample_redo.ipynb") + _code("M3_upsample_redo.ipynb")
    assert "c - 1 for c" not in code
    assert "cp = cp - 1" not in code
    assert "cp < n-1" not in code
    assert "cp[(cp > 0) & (cp < n)]" in code


def test_redo_notebooks_persist_uniform_status_and_raw_estimates():
    for path in (
        "M1_upsample_redo.ipynb",
        "M2_upsample_redo.ipynb",
        "M3_upsample_redo.ipynb",
    ):
        code = _code(path)
        assert "_evaluate_changepoints(cp, true_chgpts, n)" in code
        assert "raw_cp_{method_name}" in code
        assert "status_{method_name}" in code
        assert "results['trial_records'] = parallel_results" in code
        assert "_summarize_setting(" in code
        assert "closest_point_distance(cp, true_chgpts)" not in code
