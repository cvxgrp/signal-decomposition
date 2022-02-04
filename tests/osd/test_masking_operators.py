import unittest
import numpy as np
from osd.masking import Mask

class TestMaskScalar(unittest.TestCase):
    """
    Uses this as a standard test data set:

    np.array([ 0.,  1., nan,  3., nan, nan, nan,  7.,  8.,  9., 10., 11., 12.,
              13., nan])
    """
    def test_mask(self):
        np.random.seed(1)
        data = np.arange(15, dtype=float)
        data[np.random.uniform(size=15) < 0.2] = np.nan
        mask = Mask(~np.isnan(data))
        u = mask.mask(data)
        q = len(u)
        # test that the length is correct
        np.testing.assert_equal(np.sum(~np.isnan(data)), q)
        # test that the values are correct
        actual = np.array([ 0.,  1.,  3.,  7.,  8.,  9., 10., 11., 12., 13.])
        np.testing.assert_equal(u, actual)

    def test_unmask(self):
        np.random.seed(1)
        data = np.arange(15, dtype=float)
        data[np.random.uniform(size=15) < 0.2] = np.nan
        mask = Mask(~np.isnan(data))
        u = np.array([ 0.,  1.,  3.,  7.,  8.,  9., 10., 11., 12., 13.])
        Mtu = mask.unmask(u)
        # test that the unknown set has been replaced with zeros
        np.testing.assert_array_equal(Mtu[~mask.use_set], np.zeros(5))
        # test that the known set is equal to the input data
        np.testing.assert_array_equal(Mtu[mask.use_set], data[mask.use_set])

    def test_zerofill(self):
        np.random.seed(1)
        data = np.arange(15, dtype=float)
        data[np.random.uniform(size=15) < 0.2] = np.nan
        mask = Mask(~np.isnan(data))
        zf = mask.zero_fill(data)
        # test that the unknown set has been replaced with zeros
        np.testing.assert_array_equal(zf[~mask.use_set], np.zeros(5))
        # test that the known set is equal to the input data
        np.testing.assert_array_equal(zf[mask.use_set], data[mask.use_set])


class TestMaskVector(unittest.TestCase):
    """
    Uses this as a standard test data set:

    np.array([[ 0.,  5., nan],
              [nan, nan, nan],
              [ 2.,  7., 12.],
              [nan,  8., 13.],
              [ 4.,  9., 14.]])

    Masking operator should should return column-wise vector with nans removed
    """
    def test_mask(self):
        data = np.arange(15, dtype=float).reshape((5, 3), order='F')
        data[1] = np.nan
        data[0, -1] = np.nan
        data[3, 0] = np.nan
        mask = Mask(~np.isnan(data))
        u = mask.mask(data)
        q = len(u)
        # test that the length is correct
        np.testing.assert_equal(np.sum(~np.isnan(data)), q)
        # test that the values are correct
        actual = np.array([ 0.,  2.,  4.,  5.,  7.,  8.,  9., 12., 13., 14.])
        np.testing.assert_equal(u, actual)

    def test_unmask(self):
        data = np.arange(15, dtype=float).reshape((5, 3), order='F')
        data[1] = np.nan
        data[0, -1] = np.nan
        data[3, 0] = np.nan
        mask = Mask(~np.isnan(data))
        u = np.array([ 0.,  2.,  4.,  5.,  7.,  8.,  9., 12., 13., 14.])
        Mtu = mask.unmask(u)
        # test that the unknown set has been replaced with zeros
        np.testing.assert_array_equal(Mtu[~mask.use_set], np.zeros(5))
        # test that the known set is equal to the input data
        np.testing.assert_array_equal(Mtu[mask.use_set], data[mask.use_set])

    def test_zerofill(self):
        data = np.arange(15, dtype=float).reshape((5, 3), order='F')
        data[1] = np.nan
        data[0, -1] = np.nan
        data[3, 0] = np.nan
        mask = Mask(~np.isnan(data))
        zf = mask.zero_fill(data)
        # test that the unknown set has been replaced with zeros
        np.testing.assert_array_equal(zf[~mask.use_set], np.zeros(5))
        # test that the known set is equal to the input data
        np.testing.assert_array_equal(zf[mask.use_set], data[mask.use_set])
