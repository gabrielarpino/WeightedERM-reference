import copy

import nbformat

from scripts.m3_priority_notebook import merge_priority_outputs, priority_notebook


def _cell(source, marker):
    cell = nbformat.v4.new_code_cell(source)
    cell.metadata["marker"] = marker
    return cell


def test_priority_notebook_contains_only_base_experiment_and_plot():
    notebook = nbformat.v4.new_notebook(
        cells=[_cell(f"cell {index}", index) for index in range(11)]
    )

    staged = priority_notebook(notebook)

    assert len(staged.cells) == 5
    assert [cell.source for cell in staged.cells] == [f"cell {i}" for i in range(5)]


def test_merge_replaces_only_base_cells_and_preserves_later_experiments():
    target = nbformat.v4.new_notebook(
        cells=[_cell(f"cell {index}", index) for index in range(11)]
    )
    executed = priority_notebook(target)
    executed.cells[2].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="priority complete\n")
    ]
    later_cells = copy.deepcopy(target.cells[5:])

    merged = merge_priority_outputs(target, executed)

    assert merged.cells[2].outputs[0].text == "priority complete\n"
    assert merged.cells[5:] == later_cells


def test_merge_rejects_changed_priority_source():
    target = nbformat.v4.new_notebook(
        cells=[_cell(f"cell {index}", index) for index in range(11)]
    )
    executed = priority_notebook(target)
    executed.cells[2].source = "unexpected experiment"

    try:
        merge_priority_outputs(target, executed)
    except ValueError as error:
        assert "source mismatch" in str(error)
    else:
        raise AssertionError("merge accepted an unrelated executed notebook")
