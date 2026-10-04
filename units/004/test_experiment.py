"""Known-number and failure checks; assertion tests use ordinary Python mode."""
import numpy as np
from experiment import load_data, array_predict, loop_predict, strict_mae, channel_marker


def run_tests():
    names = []
    _, X, y = load_data()
    w = np.array([2.0, 0.1])
    weighted, combined, p = array_predict(X, w, 1)
    assert weighted.shape == (3, 2) and combined.shape == (3,) and p.shape == (3,)
    np.testing.assert_allclose(weighted, [[2, 1], [4, 2], [6, 3]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(combined, [3, 6, 9], rtol=0, atol=1e-12)
    np.testing.assert_allclose(p, [4, 7, 10], rtol=0, atol=1e-12)
    np.testing.assert_allclose(loop_predict(X, w, 1), p, rtol=0, atol=1e-12)
    assert abs(strict_mae(y, p) - 2 / 3) < 1e-12
    names.append("hand-calculated intermediate arrays and loop equivalence")
    np.testing.assert_allclose(weighted.sum(axis=0), [12, 6], rtol=0, atol=1e-12)
    assert weighted.sum(axis=1, keepdims=True).shape == (3, 1)
    assert weighted.mean() == 3
    names.append("reduction axes and keepdims")
    wrong = p.reshape(3, 1) - y
    assert wrong.shape == (3, 3)
    np.testing.assert_allclose(wrong, [[-1, -3, -5], [2, 0, -2], [5, 3, 1]], rtol=0, atol=1e-12)
    assert abs(np.abs(wrong).mean() - 22 / 9) < 1e-12
    for actual, predicted in [(y, p.reshape(3, 1)), (y, p[:2]), (np.array([]), np.array([])), (y, np.array([4, np.nan, 10]))]:
        try:
            strict_mae(actual, predicted)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid metric shape/content accepted")
    names.append("silent pairwise bug reproduced and strict metric rejects invalid inputs")
    for bad in [X.T, X.reshape(6), np.empty((0, 2)), np.array([[1, np.inf]])]:
        try:
            array_predict(bad, w, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid feature shape accepted")
    names.append("feature shape and finite-value checks")
    try:
        X + np.array([1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("Incompatible broadcast did not fail")
    np.testing.assert_allclose(X + np.array([[1], [2], [3]]), [[2, 11], [4, 22], [6, 33]])
    names.append("broadcast compatibility and sample-specific offsets")
    W = np.array([[2, -1], [0.1, 0.2]])
    np.testing.assert_allclose(X @ W, [[3, 1], [6, 2], [9, 3]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(X @ W + [1, -1], [[4, 0], [7, 1], [10, 2]], rtol=0, atol=1e-12)
    assert X.reshape(2, 3).shape == X.T.shape
    assert not np.array_equal(X.reshape(2, 3), X.T)
    np.testing.assert_array_equal(X.T, [[1, 2, 3], [10, 20, 30]])
    names.append("matrix multiplication and reshape versus transpose")
    isolated = X.copy()
    view = isolated[:, 0]
    view[0] = 99
    assert isolated[0, 0] == 99 and X[0, 0] == 1
    independent = isolated[:, 0].copy()
    independent[0] = -5
    assert isolated[0, 0] == 99
    names.append("basic slice view and explicit copy")
    nchw, nhwc = channel_marker()
    assert nchw.shape == (2, 3, 2, 2) and nhwc.shape == (2, 2, 2, 3)
    for n in range(2):
        for c in range(3):
            for h in range(2):
                for width in range(2):
                    assert nchw[n, c, h, width] == nhwc[n, h, width, c]
    names.append("all 24 unique channel markers retain identity")
    return {"status": "passed", "test_groups": names}


if __name__ == "__main__":
    import json
    print(json.dumps(run_tests(), indent=2))
