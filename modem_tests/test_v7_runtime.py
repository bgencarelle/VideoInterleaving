"""Standalone modem sessions constrain BLAS workers and restore host settings."""
import os
import unittest
from unittest import mock

import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits

from tools.v7_runtime import blas_budget


def pools():
    return {pool['filepath']: pool['num_threads'] for pool in threadpool_info()
            if pool['user_api'] == 'blas'}


class RuntimeBudgetTests(unittest.TestCase):
    def test_budget_covers_loaded_blas_and_restores_after_failure(self):
        with threadpool_limits(limits=2, user_api='blas'):
            before = pools()
            if not before:
                self.skipTest('No BLAS pool supported by threadpoolctl')
            with mock.patch.dict(os.environ, {'V7_BLAS_THREADS': '1'}):
                with self.assertRaisesRegex(RuntimeError, 'session failed'):
                    with blas_budget():
                        self.assertTrue(all(value == 1 for value in pools().values()))
                        raise RuntimeError('session failed')
            self.assertEqual(pools(), before)

    def test_opt_out_leaves_existing_library_configuration(self):
        with threadpool_limits(limits=2, user_api='blas'):
            before = pools()
            with mock.patch.dict(os.environ, {'V7_BLAS_THREADS': '0'}):
                with blas_budget():
                    self.assertEqual(pools(), before)
            self.assertEqual(pools(), before)

    def test_bad_budget_is_reported_before_session_starts(self):
        for value in ('-1', 'many', '1.5'):
            with self.subTest(value=value), mock.patch.dict(
                    os.environ, {'V7_BLAS_THREADS': value}):
                with self.assertRaisesRegex(ValueError, 'non-negative integer'):
                    with blas_budget():
                        self.fail('invalid budget entered the session')


if __name__ == '__main__':
    unittest.main()
