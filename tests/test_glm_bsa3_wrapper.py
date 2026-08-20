import WERM.comparison as comparison


def test_glm_bsa_adapters_resolve_distinct_packages():
    assert comparison.glm_package.__rname__ == "myglm2"
    assert comparison.glm3_package.__rname__ == "myglm3"
    assert comparison.glm_bsa2 is not comparison.glm_bsa3
