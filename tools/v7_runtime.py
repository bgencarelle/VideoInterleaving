"""Numerical worker budget for V7 audio sessions, standalone or application."""
from contextlib import contextmanager
import os


@contextmanager
def blas_budget():
    """Keep small per-packet BLAS products from spinning extra CPU workers.

    The limit covers the session's producer/decode threads and is restored
    afterwards. Numba's separate parallel-kernel budget is not changed.
    V7_BLAS_THREADS=0 retains the numerical libraries' own configuration.
    """
    try:
        threads = int(os.environ.get('V7_BLAS_THREADS', '1'))
    except ValueError as exc:
        raise ValueError('V7_BLAS_THREADS must be a non-negative integer') from exc
    if threads < 0:
        raise ValueError('V7_BLAS_THREADS must be a non-negative integer')
    if not threads:
        yield
        return

    # Numba's np.dot uses SciPy's BLAS. Load it before enumerating pools so
    # lazily compiled kernels cannot introduce an uncapped second library.
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads, user_api='blas'):
        yield
