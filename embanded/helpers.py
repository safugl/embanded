"""Functions used to checking input and for printing messages."""

from typing import List


def check_input(F, y,
                multi_dimensional: bool) -> None:
    """Check F and y.

    Parameters
    ----------
    F : List[np.ndarray | torch.Tensor]
        The input feature set.
    y : np.ndarray | torch.Tensor
        The target regressor.
    multi_dimensional : bool
        Whether multi_dimensional has been specified.
    """
    dtype = y.dtype

    # First check F
    if isinstance(F, list) is not True:
        raise TypeError(r"{F} must be a list")

    # Then check that F and y has same number of rows
    num_rows = y.shape[0]
    for X_f in F:
        if X_f.shape[0] != num_rows:
            raise TypeError(r"Each array should have {num_rows} rows")
        if X_f.dtype != dtype:
            raise TypeError(r"Expected dtype {dtype}")

    # y must have 2 dimensions
    if y.ndim != 2:
        raise TypeError(r"y must have two dimensions, not {y.ndim}")

    # Then check if y has multiple columns
    if multi_dimensional is False:
        if y.shape[-1] != 1:
            raise TypeError(r"y must have 1 column, not {y.ndim}")


def check_positive_float(val: float) -> None:
    """Check if value is positive float."""
    assert isinstance(val, float), f'{val} should be floats.'
    assert val > 0, f'{val} should be positive.'


def check_boolean(val: bool) -> None:
    """Check if val is a boolean variable."""
    if not isinstance(val, bool):
        raise ValueError(f'{val} should be a boolean variable.')


def check_smooth_params(vals: List[float | None]) -> None:
    """Check the smoothness parameters."""
    if not isinstance(vals, list):
        raise ValueError(f'{vals} should be a list.')

    for val in vals:
        if val is not None:
            check_positive_float(val)
