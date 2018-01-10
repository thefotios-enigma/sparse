import numpy as np
import scipy.sparse
from .core import COO


def allclose(
    x,
    y,
    maybe_canonicalize=True,
    maybe_densify=True,
    rtol=1e-05,
    atol=1e-08,
):
    """ Efficiently two arrays, both sparse and dense

    Parameters
    ----------
    x, y: {COO, scipy.sparse.spmatrix, np.ndarray}
        The arrays to be compared
    maybe_canonicalize: bool, optional
        Canonicalize COO arrays for efficient comparison
    maybe_densify: bool, optional
        Densify COO arrays if sparse comparison fails. Needs to be
        :code:`True` when comparing dense and sparse arrays
    rtol: float
        Relative tolerance
    atol: float
        Absolute tolerance

    See Also
    --------
    numpy.allclose : NumPy equivalent, also used internally

    """
    xt = type(x)
    yt = type(y)

    # Trivial rejects
    if not x.shape == y.shape:
        return False
    if not x.dtype == y.dtype:
        return False

    if isinstance(x, scipy.sparse.spmatrix):
        x = COO.from_scipy_sparse(x)

    if isinstance(y, scipy.sparse.spmatrix):
        y = COO.from_scipy_sparse(y)

    # Both COO, so compare their attributes
    if isinstance(x, COO) and isinstance(y, COO):
        if maybe_canonicalize:
            x.sum_duplicates()
            y.sum_duplicates()

            if not x.nnz == y.nnz:
                return False

        # Both canonical
        if (
            x.sorted and y.sorted and
            is_lexsorted(x) and is_lexsorted(y) and
            not x.has_duplicates and not y.has_duplicates
        ):
            return (
                np.array_equal(x.coords, y.coords) and
                np.allclose(x.data, y.data, rtol=rtol, atol=atol)
            )

    # Data seems very heterogenous, let's try densifying, if we are allowed
    if maybe_densify:
        try:
            xx = x.maybe_densify()
        except AttributeError:
            xx = x
        try:
            yy = y.maybe_densify()
        except AttributeError:
            yy = y
        if np.allclose(xx, yy, rtol=rtol, atol=atol):
            return True

        # Just blindly try numpy allclose
        try:
            return np.allclose(x, y, rtol=rtol, atol=atol)
        except TypeError:
            pass

    # None of the above worked
    raise ValueError(
        "Trying to compare two arrays of types %s and %s, but none of the "
        "checks could be applied." % (xt, yt)
    )


def is_lexsorted(x):
    return not x.shape or (np.diff(x.linear_loc()) > 0).all()
