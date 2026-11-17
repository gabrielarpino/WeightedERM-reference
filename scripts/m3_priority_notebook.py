"""Stage and merge self-contained M3 notebook subsets for serial execution."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat


BASE_CELL_COUNT = 5


def priority_notebook(notebook):
    """Copy the base M3d experiment and its plot into an executable notebook."""
    if len(notebook.cells) < BASE_CELL_COUNT:
        raise ValueError("M3 notebook does not contain the five base cells")
    staged = copy.deepcopy(notebook)
    staged.cells = copy.deepcopy(notebook.cells[:BASE_CELL_COUNT])
    return staged


def alpha_notebook(notebook):
    """Copy the appended alpha experiments into an executable notebook."""
    if len(notebook.cells) <= BASE_CELL_COUNT:
        raise ValueError("M3 notebook does not contain appended alpha cells")
    staged = copy.deepcopy(notebook)
    staged.cells = copy.deepcopy(notebook.cells[BASE_CELL_COUNT:])
    return staged


def merge_priority_outputs(target, executed):
    """Merge executed base cells while preserving all later experiment cells."""
    if len(executed.cells) != BASE_CELL_COUNT:
        raise ValueError("executed priority notebook must contain exactly five cells")
    if len(target.cells) < BASE_CELL_COUNT:
        raise ValueError("target M3 notebook does not contain five base cells")
    for index, (current, completed) in enumerate(
        zip(target.cells[:BASE_CELL_COUNT], executed.cells)
    ):
        if current.source != completed.source:
            raise ValueError(f"source mismatch in priority cell {index}")
    merged = copy.deepcopy(target)
    merged.cells[:BASE_CELL_COUNT] = copy.deepcopy(executed.cells)
    return merged


def _read(path: Path):
    return nbformat.read(path, as_version=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("stage-priority", "stage-alpha"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("source", type=Path)
        subparser.add_argument("output", type=Path)

    merge_parser = subparsers.add_parser("merge-priority")
    merge_parser.add_argument("target", type=Path)
    merge_parser.add_argument("executed", type=Path)

    args = parser.parse_args()
    if args.command == "stage-priority":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(priority_notebook(_read(args.source)), args.output)
    elif args.command == "stage-alpha":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(alpha_notebook(_read(args.source)), args.output)
    else:
        nbformat.write(
            merge_priority_outputs(_read(args.target), _read(args.executed)),
            args.target,
        )


if __name__ == "__main__":
    main()
