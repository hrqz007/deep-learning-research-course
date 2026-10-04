"""Independent Fraction triple-loop oracle, exact hand fixtures and error tests."""
from fractions import Fraction
import json
import numpy as np
from experiment import affine, linear, compose_affine, load_case, run, broadcast_evidence


def fraction_rows(rows):
    return [[v if isinstance(v, Fraction) else Fraction(str(v)) for v in row] for row in rows]


def exact_product(A, B):
    A, B = fraction_rows(A), fraction_rows(B)
    if len(A[0]) != len(B):
        raise ValueError("incompatible exact product")
    # Independent Python loops; no NumPy @, dot, sum-reduction, or experiment helper.
    output = []
    for i in range(len(A)):
        row = []
        for k in range(len(B[0])):
            total = Fraction(0)
            for j in range(len(B)):
                total += A[i][j] * B[j][k]
            row.append(total)
        output.append(row)
    return output


def exact_affine(A, W, b):
    product = exact_product(A, W)
    offsets = [v if isinstance(v, Fraction) else Fraction(str(v)) for v in b]
    return [[value + offsets[k] for k, value in enumerate(row)] for row in product]


def floats(A):
    return np.array([[float(v) for v in row] for row in A], dtype=float)


def same(actual, expected):
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def run_tests():
    groups = []
    data, a = load_case()
    X, W1, b1, W2, b2 = (a[k] for k in ("X", "W1", "b1", "W2", "b2"))
    same(linear(X, W1), [[5,2,1],[-2,0,-1],[5,6,-2]])
    U = affine(X, W1, b1)
    same(U, [[6,0,1.5],[-1,-2,-.5],[6,4,-1.5]])
    Y = affine(U, W2, b2)
    same(Y, [[2,4.5],[-2,-1.5],[10,9.5]])
    Wc, bc = compose_affine(W1, b1, W2, b2)
    same(Wc, [[4,3],[0,1]]); same(bc, [-2,-.5]); same(affine(X,Wc,bc), Y)
    Hq = exact_affine(data["X"], data["W1"], data["b1"])
    Yq = exact_affine(Hq, data["W2"], data["b2"])
    same(U, floats(Hq)); same(Y, floats(Yq))
    groups.append("central hand arrays and independent Fraction oracle")

    rng = np.random.default_rng(9009)
    fixtures = 0
    for N in [1,2,3,7]:
        for D,Hwidth,K in [(1,1,1),(2,3,2),(3,2,4),(4,5,1)]:
            for denominator in [1,3,5,7]:
                def draw(rows, cols):
                    return [[Fraction(int(v), denominator) for v in row]
                            for row in rng.integers(-5, 6, size=(rows, cols))]
                Xq,W1q,W2q = draw(N,D), draw(D,Hwidth), draw(Hwidth,K)
                b1q,b2q = draw(1,Hwidth)[0],draw(1,K)[0]
                Xf,W1f,W2f = floats(Xq),floats(W1q),floats(W2q)
                b1f,b2f = np.array([float(v) for v in b1q]),np.array([float(v) for v in b2q])
                expected = floats(exact_affine(exact_affine(Xq,W1q,b1q),W2q,b2q))
                sequential = affine(affine(Xf,W1f,b1f),W2f,b2f)
                fusedW, fusedb = compose_affine(W1f,b1f,W2f,b2f)
                same(sequential, expected); same(affine(Xf,fusedW,fusedb), expected)
                same(linear(linear(Xf,W1f),W2f), linear(Xf,linear(W1f,W2f)))
                same((Xf@W1f).T, W1f.T@Xf.T)
                fixtures += 1
    groups.append("64 rational fixtures across non-square shapes and N=1")

    same(linear(np.eye(2), Wc), Wc)
    u,v = np.array([[2.,1.]]),np.array([[-1.,3.]])
    same(linear(2*u-3*v,Wc),2*linear(u,Wc)-3*linear(v,Wc))
    same(affine((1-.3)*u+.3*v,Wc,bc),(1-.3)*affine(u,Wc,bc)+.3*affine(v,Wc,bc))
    same(affine(u,Wc,bc)-affine(v,Wc,bc),linear(u-v,Wc))
    assert not np.allclose(affine(2*u,Wc,bc),2*affine(u,Wc,bc))
    same(affine(np.zeros((1,2)),Wc,bc),[[-2,-.5]])
    groups.append("basis rows, linearity, affine combination, origin and differences")

    S = np.diag([2.,1.]); T = np.array([[1.,1.],[0.,1.]])
    same(S@T,[[2,2],[0,1]]); same(T@S,[[2,1],[0,1]])
    assert not np.array_equal(S@T,T@S)
    point=np.array([[1.,1.]])
    same(point@S@S.T,[[4,1]]); same(point@S@np.diag([.5,1.]),point)
    same(X.T@X,[[10,5],[5,6]])
    assert np.asarray([1,2]).T.shape == (2,)
    groups.append("noncommutativity, transpose is not inverse, wrong-axis aggregation")

    evidence = broadcast_evidence(b1,3)
    assert evidence["same_values"] and evidence["broadcast_shares_memory"]
    assert not evidence["tile_shares_memory"] and not evidence["broadcast_writeable"]
    same(affine(X,W1,b1[None,:]),U)
    wrong = X@W1 + b1[:,None]
    same(wrong,[[6,3,2],[-4,-2,-3],[5.5,6.5,-1.5]])
    assert wrong.shape == U.shape and not np.allclose(wrong,U)
    rejects(lambda: affine(X,W1,b1[:,None]))
    groups.append("shared bias broadcasting, explicit copy, square-batch silent error")

    changed=W1.copy(); changed[0,1] += .25
    same(linear(X,changed)-linear(X,W1),[[0,.25,0],[0,0,0],[0,.75,0]])
    same(affine(X,W1,b1+np.array([0,.25,0]))-U,[[0,.25,0]]*3)
    same(affine(X[[2,0,1]],W1,b1), U[[2,0,1]])
    same(affine(X[0:1],W1,b1), U[0:1])
    groups.append("parameter changes, shared bias shifts and row independence")

    rejects(lambda: affine(X[0],W1,b1))
    rejects(lambda: affine(X,W1.T,b1))
    rejects(lambda: affine(X,W1,[1,2]))
    rejects(lambda: affine(X,W1,1))
    rejects(lambda: affine(X,W1,np.zeros((3,3))))
    rejects(lambda: linear(np.zeros((0,2)),W1))
    rejects(lambda: linear(np.zeros((2,0)),np.zeros((0,3))))
    rejects(lambda: linear(np.full((2,2),np.nan),W1))
    rejects(lambda: affine(X,W1,[1,np.inf,0]))
    rejects(lambda: affine(X,W1,np.array([1,2,3],dtype=complex)),TypeError)
    rejects(lambda: linear([["1","2"]],W1),TypeError)
    rejects(lambda: linear([[True,False]],W1),TypeError)
    rejects(lambda: linear([[True,1]],[[1],[2]]),TypeError)
    rejects(lambda: linear(((np.bool_(True),1),),[[1],[2]]),TypeError)
    rejects(lambda: linear([[1,1]],[[1],[False]]),TypeError)
    rejects(lambda: linear([[1,1]],[[1],[np.bool_(True)]]),TypeError)
    rejects(lambda: affine([[1,1]],np.eye(2),[True,.5]),TypeError)
    rejects(lambda: affine([[1,1]],np.eye(2),(np.bool_(True),.5)),TypeError)
    rejects(lambda: compose_affine(W1,b1,np.ones((2,2)),b2))
    rejects(lambda: linear([[1e308]],[[1e308]]),FloatingPointError)
    # Once NumPy has already promoted a mixed bool/integer array, history is gone.
    same(linear(np.array([[True,1]]),[[1],[2]]),[[3]])
    groups.append("20 rejected shape, type, mixed-boolean, finite-input and overflow cases")
    return {"status":"passed","groups":groups,"rational_fixtures":fixtures,
            "exact_Y_mm":[[str(v) for v in row] for row in Yq],
            "rtol":1e-12,"atol":1e-12}


if __name__ == "__main__":
    print(json.dumps(run_tests(),ensure_ascii=False,indent=2))
