import numpy as np
import pytest

from WERM import comparison as c


@pytest.mark.parametrize('lambda_value', [0.05, 'min'])
def test_serial_native_moseg_matches_original_socket_algorithm(lambda_value):
    rng = np.random.default_rng(88)
    X = rng.normal(size=(120, 8))
    beta = np.zeros((120, 8))
    beta[:40, 0] = 3
    beta[40:80, 1] = 3
    beta[80:, 2] = 3
    y = np.sum(X * beta, axis=1) + rng.normal(scale=0.3, size=120)
    with c.conversion.localconverter(c.converter):
        c.ro.r('set.seed(47)')
        original = c.moseg_package.moseg(
            X, y, G=20, n_cps=2, do_plot=False, **{'lambda': lambda_value}
        )
        c.ro.r('set.seed(47)')
        serial = c._moseg_native_serial(
            X, y, G=20, n_cps=2, do_plot=False, **{'lambda': lambda_value}
        )
    for index in (0, 1, 2, 3, 4):
        np.testing.assert_allclose(np.asarray(serial[index]), np.asarray(original[index]),
                                   rtol=1e-10, atol=1e-10)


def test_native_serial_never_opens_cluster_even_when_default_port_is_occupied():
    # Binding the exact default port makes the old implementation fail, while
    # the sequential inner backend must remain independent of socket resources.
    import socket
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', 0))
        sock.listen()
        port = sock.getsockname()[1]
        c.ro.r(f'parallel:::setDefaultClusterOptions(port={port})')
        try:
            with c.conversion.localconverter(c.converter):
                rng = np.random.default_rng(5)
                X = rng.normal(size=(80, 5))
                y = X[:, 0] + rng.normal(size=80)
                result = c._moseg_native_serial(
                    X, y, G=15, n_cps=2, do_plot=False, **{'lambda': 0.1}
                )
                assert len(result) >= 3
        finally:
            c.ro.r('parallel:::setDefaultClusterOptions(port=11000L)')
