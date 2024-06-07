"""
This module contains functions to create basis and regularization matrices for
a time series with one or multiple periods.
:author: Bennet Meyers, Mehmet Giray, Aramis Dufour
"""

import numpy as np
from scipy.sparse import spdiags
from itertools import combinations


def make_basis_matrix(
    num_harmonics,
    length,
    periods,
    standing_wave=False,
    trend=False,
    max_cross_k=None,
    custom_basis=None,
):
    """
    This function creates a basis matrix for a time series with one or multiple
    periods. It creates a Fourier basis matrix for each period, then combines
    them into a single matrix, including cross-terms. It also includes an offset
    term, and can include a linear trend.

    :param num_harmonics: number of harmonics to use for each period.
    :type num_harmonics: int or 1D iterable of ints
    :param length: length of the time series.
    :type length: int
    :param periods: periods of the signal, each period yields a Fourier basis matrix.
    :type periods: float or 1D iterable of floats
    :param standing_wave: whether to use the standing wave basis for each period.
        A standing wave basis only includes sine terms and starts at T/2. Defaults
        to False.
    :type standing_wave: bool or 1D iterable of bools
    :param trend: whether to include a linear trend in the basis matrix. Defaults to
        False.
    :type trend: bool
    :param max_cross_k: maximum number of cross terms. Defaults to None.
    :type max_cross_k: int
    :param custom_basis: allows the user to pass a dictionary of custom basis
        matrices. The key is the index of the period and the value is the basis
        matrix. The user can input a shorter basis than length, but the associated
        period must be an int. Defaults to None.
    :type custom_basis: dict
    :return: basis matrix, it's shape depends on the parameters.
    :rtype 2D array of floats
    """
    sort_idx, Ps, num_harmonics, standing_wave = initialize_arrays(
        num_harmonics, periods, standing_wave, custom_basis
    )
    # Make the basis
    t_values = np.arange(length)  # Time stamps (row vector)
    B_fourier = []
    for ix, P in enumerate(Ps):
        i_values = np.arange(1, num_harmonics[ix] + 1)[
            :, np.newaxis
        ]  # Harmonic indices (column vector)
        if standing_wave[ix]:
            w = 2 * np.pi / (P * 2)
            B_sin = np.sin(i_values * w * np.mod(t_values, P))
            B_f = np.empty((length, num_harmonics[ix]), dtype=float)
            B_f[:] = B_sin.T
        else:
            w = 2 * np.pi / P
            B_cos = np.cos(i_values * w * t_values)
            B_sin = np.sin(i_values * w * t_values)
            B_f = np.empty((length, 2 * num_harmonics[ix]), dtype=float)
            B_f[:, ::2] = B_cos.T
            B_f[:, 1::2] = B_sin.T
        B_fourier.append(B_f)
    # Use custom basis if provided
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            # check length
            if val.shape[0] != length:
                # extend to cover future time period if necessary
                multiplier = max(1, length // val.shape[0] + 1)
                new_val = np.tile(val, (multiplier, 1))[:length]
            else:
                new_val = val[:length]
            # also reorder index of custom basis, if necessary
            ixt = np.where(sort_idx == ix)[0][0]
            B_fourier[ixt] = new_val
    # Add offset and linear terms
    if trend is False:
        B_P0 = np.ones((length, 1))
        B0 = [B_P0]
    else:
        v = np.sqrt(3)
        B_PL = np.linspace(-v, v, length).reshape(-1, 1)
        B_P0 = np.ones((length, 1))
        B0 = [B_PL, B_P0]
    # Cross terms, this handles the case of no cross terms gracefully
    # (empty list)
    C = [
        cross_bases(*base_tuple, max_k=max_cross_k)
        for base_tuple in combinations(B_fourier, 2)
    ]
    B_list = B0 + B_fourier + C
    B = np.hstack(B_list)
    return B


def make_regularization_matrix(
    num_harmonics,
    weight,
    periods,
    standing_wave=False,
    trend=False,
    max_cross_k=None,
    custom_basis=None,
):
    """
    This function creates a regularization matrix for a time series with one or
    multiple periods. It creates a regularization matrix for each period, then
    combines them into a single matrix. The weights are determined using the
    Dirichlet energy of the basis functions.

    :param num_harmonics: number of harmonics to use for each period.
    :type num_harmonics: int or 1D iterable of ints
    :param weight: weight for the regularization matrix.
    :type weight: float
    :param periods: periods of signal, each period will yield a bloc.
    :type periods: float or 1D iterable of floats
    :param standing_wave: whether to use the standing wave basis for each period.
    :type standing_wave: bool or 1D iterable of bools
    :param trend: whether to include a linear trend in the basis matrix. Defaults
        to False.
    :type trend: bool
    :param max_cross_k: maximum number of cross terms. Defaults to None.
    :type max_cross_k: int
    :param custom_basis: allows the user to pass a dictionary of custom basis
        matrices.
    :type custom_basis: dict
    :return: regularization matrix, it's shape depends on the parameters.
    :rtype 2D array of floats
    """
    sort_idx, Ps, num_harmonics, standing_wave = initialize_arrays(
        num_harmonics, periods, standing_wave, custom_basis
    )
    ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]
    # Create a sequence of values from 1 to K (repeated for cosine
    # and sine when not standing wave)
    i_value_list = []
    for ix, nh in enumerate(num_harmonics):
        if standing_wave[ix]:
            i_value_list.append(np.arange(1, nh + 1))
        else:
            i_value_list.append(np.repeat(np.arange(1, nh + 1), 2))
    # Create blocks of coefficients
    blocks_original = [iv * lx for iv, lx in zip(i_value_list, ls_original)]
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            ixt = np.where(sort_idx == ix)[0][0]
            blocks_original[ixt] = ls_original[ixt] * np.arange(1, val.shape[1] + 1)
    if max_cross_k is not None:
        max_cross_k *= 2
    # This assumes  that the list of periods is ordered, which is ensured
    # when calling initialize_arrays.
    blocks_cross = [
        [l2 for l1 in c[0][:max_cross_k] for l2 in c[1][:max_cross_k]]
        for c in combinations(blocks_original, 2)
    ]
    # Combine the blocks to form the coefficient array
    if trend is False:
        first_block = [np.zeros(1)]
    else:
        first_block = [np.zeros(2)]
    coeff_i = np.concatenate(first_block + blocks_original + blocks_cross)
    # Create the diagonal matrix
    D = spdiags(coeff_i, 0, coeff_i.size, coeff_i.size)
    return D


def initialize_arrays(num_harmonics, periods, standing_wave, custom_basis):
    """
    This function initializes the arrays for periods, harmonics, and standing wave.
    It sorts everything in descending order of periods. It also performs some basic
    checks on the input parameters.

    :param num_harmonics: number of harmonics to use for each period.
    :type num_harmonics: int or 1D iterable of ints
    :param periods: periods of the signal, each period yields a bloc.
    :type periods: float or 1D iterable of floats
    :param standing_wave: whether to use standing wave basis for each period.
    :type standing_wave: bool or 1D iterable of bools
    :param custom_basis: allows the user to pass a dictionary of custom basis
        matrices.
    :type custom_basis: dict
    :raises TypeError: raised if custom_basis is not a dictionary or None.
    :raises ValueError: raised if num_harmonics and periods have different lengths.
    :raises ValueError: raised if standing_wave and periods have different lengths.
    :return: tuple with sorted indices, sorted periods, sorted number of harmonics, sorted
        standing wave.
    :rtype tuple
    """
    # Custom basis
    if not (isinstance(custom_basis, dict) or custom_basis is None):
        raise TypeError(
            "custom_basis should be a dictionary where the key is the index\n"
            + "of the period and the value is list containing the basis and the weights"
        )
    # Periods
    Ps = np.atleast_1d(periods)
    # Number of harmonics
    num_harmonics = np.atleast_1d(num_harmonics)
    if len(num_harmonics) == 1:
        num_harmonics = np.tile(num_harmonics, len(Ps))
    elif len(num_harmonics) != len(Ps):
        raise ValueError(
            "Please pass a single number of harmonics for all periods or a number\n"
            + "for each period"
        )
    # Standing wave
    standing_wave = np.atleast_1d(standing_wave)
    if len(standing_wave) == 1:
        standing_wave = np.tile(standing_wave, len(Ps))
    elif len(standing_wave) != len(Ps):
        raise ValueError(
            "Please pass a single boolean for standing_wave for all periods or a\n"
            + "boolean for each period"
        )
    # Sort the periods, harmonics, and standing wave
    sort_idx = np.argsort(-Ps)
    Ps = -np.sort(-Ps)  # Sort in descending order
    num_harmonics = num_harmonics[sort_idx]
    standing_wave = standing_wave[sort_idx]
    return sort_idx, Ps, num_harmonics, standing_wave


def cross_bases(B_P1, B_P2, max_k=None):
    """
    This function computes the cross terms between two basis matrices.

    :param B_P1: basis matrix for the first period.
    :type B_P1: 2D array of floats
    :param B_P2: basis matrix for the second period.
    :type B_P2: 2D array of floats
    :param max_k: maximum number of cross terms. Defaults to None.
    :type max_k: int
    :return: cross terms matrix.
    :rtype 2D array of floats
    """
    if max_k is None:
        # Reshape both arrays to introduce a new axis for broadcasting
        B_P1_new = B_P1[:, :, None]
        B_P2_new = B_P2[:, None, :]
    else:
        B_P1_new = B_P1[:, : 2 * max_k, None]
        B_P2_new = B_P2[:, None, : 2 * max_k]
    # Use broadcasting to compute the outer product for each row
    result = B_P1_new * B_P2_new
    # Reshape the result to the desired shape
    result = result.reshape(result.shape[0], -1)
    return result


def pinball_slopes(quantiles):
    """
    This function computes the slopes for the pinball loss function.

    :param quantiles: ndarray, 1D array quantiles.
    :return: tuple, slopes.
    """
    percentiles = np.asarray(quantiles)
    a = quantiles - 0.5
    b = (0.5) * np.ones((len(a),))
    return a, b