"""
Result verification utilities.
"""

import numpy as np


def verify_results(
    result: np.ndarray,
    reference: np.ndarray,
    op_name: str = "",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    verbose: bool = False,
) -> bool:
    """
    Verify that results match reference within tolerance.

    Args:
        result: The computed result array
        reference: The reference (expected) array
        op_name: Name of the operation (for error messages)
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Whether to print warnings on mismatch

    Returns:
        True if results match, False otherwise
    """
    try:
        np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        if verbose:
            print(f"  WARNING: {op_name} results differ: {e}")
        return False
