import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "rebuttal" / "wer8_common.py"
SPEC = importlib.util.spec_from_file_location("wer8_common", MODULE_PATH)
wer8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wer8)


def test_historical_grids_and_repetition_count():
    assert wer8.REPETITIONS == 100
    assert np.array_equal(wer8.M2_DELTAS, np.arange(0.5, 16.0, 1.0))
    assert np.allclose(
        wer8.M3_KAPPAS,
        np.sqrt([1.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0, 2000.0]),
    )


def test_repetition_seeds_are_stable_and_distinct():
    seeds = [wer8.repetition_seed("m2", 0, repetition) for repetition in range(100)]
    assert seeds == [wer8.repetition_seed("m2", 0, repetition) for repetition in range(100)]
    assert len(set(seeds)) == 100
    assert wer8.repetition_seed("m2", 0, 0) != wer8.repetition_seed("m3", 0, 0)


def test_worker_cap():
    workers, available = wer8.effective_workers(10_000)
    assert 1 <= workers <= min(32, available)
