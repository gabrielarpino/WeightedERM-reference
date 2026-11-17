import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PLOTS = {
    "M3_upsample_redo2.ipynb": 3,
    "M3_upsample_redo3.ipynb": 4,
}
SELECTED = ("GLMBSA-CV", "WERM-L1-fullCV-2fold")


@pytest.mark.parametrize("notebook_name, expected_count", EXPECTED_PLOTS.items())
def test_each_experiment_has_executed_two_fold_two_method_replot(notebook_name, expected_count):
    notebook = json.loads((ROOT / notebook_name).read_text())
    cells = notebook["cells"]
    replots = [
        cell
        for cell in cells
        if cell.get("metadata", {}).get("m3_two_method_replot")
        and not cell.get("metadata", {}).get("m3_delta6_extension")
    ]

    assert len(replots) == expected_count
    for cell in replots:
        source = "".join(cell["source"])
        assert tuple(cell["metadata"]["selected_methods"]) == SELECTED
        assert source.count("'name': 'GLMBSA-CV'") == 1
        assert source.count("'name': 'WERM-L1-fullCV-2fold'") == 1
        assert "WERM-L1-fullCV-10fold" not in source
        assert "GLMBSA-CV-5fold" not in source
        assert "_two_method_2fold" in source
        assert "plot_location_asymmetric_errors(" in source
        assert "plot_size_asymmetric_errors(" in source
        assert "plot_runtime_asymmetric_errors(" in source
        assert "plot_location_boxplots(" in source
        assert "plot_size_boxplots(" in source
        assert "plot_runtime_boxplots(" in source
        assert "plot_changepoint_histograms_publication(" in source
        assert cell.get("execution_count") is not None
        assert any(output.get("output_type") == "display_data" for output in cell.get("outputs", []))

        index = cells.index(cell)
        assert index > 0
        previous_code = next(candidate for candidate in reversed(cells[:index]) if candidate["cell_type"] == "code")
        assert previous_code.get("metadata", {}).get("m3_two_method_replot") is not True
