from tools.myocardial_exact_penalty import matched_specifications, is_exact_match


def test_small_penalty_grid_is_matched_and_ordered():
    specs = matched_specifications()
    assert len(specs) == 10
    for family in ('l1', 'l2'):
        strengths = [s for f, s in specs if f == family]
        assert strengths == [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]


def test_family_only_is_not_a_match():
    assert is_exact_match('l1', 1e-6, 'l1', 1e-6)
    assert not is_exact_match('l1', 1e-4, 'l1', 1e-6)
    assert not is_exact_match('l1', 1e-6, 'l2', 1e-6)
