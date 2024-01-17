# pylint: skip-file

import copy
from absl.testing import absltest
import numpy as np
import pytest
from .embanded_numpy import EMBanded
from . import _numpy_model_utils
from . import _numpy_linalg_utils


class EMBandedTests(absltest.TestCase):
    """Test cases for EMBanded."""

    def test_compare_model_settings(self):
        """Fit models with different settings and compare outputs"""

        np.random.seed(1)
        num_obs = 400
        for num_groups in [1, 2, 4, 8, 16, 32]:
            F = [[]]*num_groups
            W = [[]]*num_groups
            for f in range(num_groups):
                F[f] = np.random.randn(num_obs, 5) + 100
                if np.random.rand(1) < 0.5:
                    W[f] = np.random.randn(5, 1)
                else:
                    W[f] = np.zeros((5, 1))

            X = np.concatenate(F, axis=1)
            W = np.concatenate(W, axis=0)
            N = np.random.randn(num_obs, 1)*2
            Y = X@W + N
            X_before = copy.deepcopy(X)
            Y_before = copy.deepcopy(Y)
            _compare_models(F, Y)

            np.testing.assert_equal(X, X_before)
            np.testing.assert_equal(Y, Y_before)

    def test_fails_hyperparas(self):
        """Tests cases for wrong hyperparameter inputs"""
        with pytest.raises(Exception):
            EMBanded(hyper_params=(-1, 1e-4, 1e-4, 1e-4))
        with pytest.raises(Exception):
            EMBanded(hyper_params=(0, 1e-4, 1e-4, 1e-4))

    def test_fails_smoothness_param(self):
        """Tests cases for wrong smoothness parameter inputs"""

        with pytest.raises(Exception):
            emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                           max_iterations=200)
            emb.set_smoothness_param(['str', None, None])

        with pytest.raises(Exception):
            emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                           max_iterations=200)
            emb.set_smoothness_param([-1, None, None])

    def test_fails_input_data1(self):
        """Tests cases for wrong input data, F and y."""

        with pytest.raises(Exception):
            # F should be a list of matrices
            emb = EMBanded()
            emb.fit(np.random.randn(100, 10), np.random.randn(100, 1))

    def test_fails_input_data2(self):
        """Tests cases for wrong input data, F and y."""
        with pytest.raises(Exception):
            # y has too many rows.
            emb = EMBanded()
            F = [np.random.randn(100, 1), np.random.randn(100, 10),
                 np.random.randn(100, 10)]
            emb.fit(F, np.random.randn(1000, 1))

    def test_fails_input_data3(self):
        """Tests cases for wrong input data, F and y."""
        with pytest.raises(Exception):
            # y has too many columns.
            emb = EMBanded()
            F = [np.random.randn(100, 1), np.random.randn(100, 10),
                 np.random.randn(100, 10)]
            emb.fit(F, np.random.randn(100, 10))

    def test_fails_initializations(self):
        """Tests cases for wrong hyper parameter intializations"""

        with pytest.raises(Exception):
            emb = EMBanded()
            emb.set_lambdas_init(1.)

        with pytest.raises(Exception):
            np.random.seed(1)
            emb = EMBanded()
            emb.set_lambdas_init(np.random.randn(100))

        with pytest.raises(Exception):
            emb = EMBanded()
            emb.set_nu_init(None)

        with pytest.raises(Exception):
            emb = EMBanded()
            emb.set_multidimensional(None)

    def test_compare_with_ols(self):
        """Fit model and compare with OLS"""

        np.random.seed(1)
        num_obs = 400

        # Simulate training data
        F = [np.random.randn(num_obs, 100), np.random.randn(num_obs, 100)]
        X = np.concatenate(F, axis=1)
        X_copy = copy.deepcopy(X)
        W = np.concatenate(
            [np.random.randn(100, 1), np.zeros((100, 1))], axis=0)
        N = np.random.randn(num_obs, 1)*5
        Y = X@W + N + 100
        emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                       max_iterations=200)
        emb.fit(F, Y)

        # Check deep copies
        np.testing.assert_array_equal(X, X_copy)
        np.testing.assert_array_equal(np.concatenate(F, axis=1), X_copy)

        X_offset = X.mean(axis=0, keepdims=True)

        # Simulate validation data
        F_val = [np.random.randn(num_obs, 100), np.random.randn(num_obs, 100)]
        X_val = np.concatenate(F_val, axis=1)
        N_val = np.random.randn(num_obs, 1)*5
        Y_val = X_val @ W + N_val + 100

        D_train = copy.deepcopy(X) - X_offset
        D_val = copy.deepcopy(X_val) - X_offset

        W_ols = np.linalg.lstsq(np.c_[np.ones(num_obs), D_train], Y,
                                rcond=None)[0]
        L_em = np.sum((emb.W-W)**2)
        L_ols = np.sum((W_ols[1:]-W)**2)

        self.assertGreater(L_ols, L_em)

        pred_ols = np.c_[np.ones(num_obs), D_val]@W_ols
        pred_emb = emb.predict(F_val)

        loss_ols = np.mean(np.abs(pred_ols-Y_val)**2)
        loss_emb = np.mean(np.abs(pred_emb-Y_val)**2)

        self.assertGreater(loss_ols, loss_emb)

    def test_Omega(self):
        """Fit model with smoothness assumptions and test Omega terms"""
        np.random.seed(1)

        num_obs = 20
        X = np.random.randn(num_obs, 10) + 1000
        Y = np.random.randn(num_obs, 1)

        emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                       max_iterations=200)
        emb.set_store_covariance_terms(True)
        emb.fit([X], Y)

        assert emb.covariance_terms['Omega'] is None
        assert emb.covariance_terms['Omega_inv'] is None

        def compare_estimates(h, A, B, mult_dim):
            """Fit the model using X=A and y=B"""
            emb_smooth = EMBanded(hyper_params=(
                1e-4, 1e-4, 1e-4, 1e-4),
                max_iterations=200)
            emb_smooth.set_smoothness_param([h])
            emb_smooth.set_store_covariance_terms(True)
            emb_smooth.set_multidimensional(mult_dim)
            emb_smooth.fit([A], B)

            x_grid = np.arange(A.shape[1])[None, ...]

            Omega_j = (
                1 + np.sqrt(3) * np.abs(x_grid.T - x_grid) / h
            ) * np.exp(-np.sqrt(3) * np.abs(x_grid.T - x_grid) / h)

            np.testing.assert_array_almost_equal(
                emb_smooth.covariance_terms['Omega'], Omega_j)
            O_i = np.linalg.inv(np.linalg.cholesky(Omega_j))
            np.testing.assert_array_almost_equal(
                emb_smooth.covariance_terms['Omega_inv'], O_i.T@O_i)

        for h in np.linspace(0.1, 100, 10):
            for mult_dim in [False, True]:
                compare_estimates(h, X, Y, mult_dim)

    def test_offset(self):
        """Test if offsets are stored apprropriately"""

        np.random.seed(1)

        num_obs = 20
        X = np.random.randn(num_obs, 10) + 100
        Y = np.random.randn(num_obs, 1)

        emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                       max_iterations=200)
        emb.set_store_covariance_terms(True)
        emb.fit([X], Y)

        np.testing.assert_array_almost_equal(
            emb.X_offset, np.mean(X, axis=0, keepdims=True))
        np.testing.assert_array_almost_equal(
            emb.y_offset, np.mean(Y, axis=0, keepdims=True))

    def test_utils_add_diag(self):
        """Test utils. matrix_add_to_diagonal"""

        X = np.random.randn(10000, 1000)
        A1 = X.T@X
        A2 = copy.deepcopy(A1)
        B = np.random.randn(A1.shape[1])
        np.testing.assert_almost_equal(
            _numpy_linalg_utils.matrix_add_to_diagonal(A1, B), A2+np.diag(B))

    def test_utils_get_diag(self):
        """Test utils. matrix_get_diagonal_elements"""

        X = np.random.randn(10000, 1000)
        A1 = X.T@X
        A2 = copy.deepcopy(A1)
        np.testing.assert_almost_equal(
            _numpy_linalg_utils.matrix_get_diagonal_elements(A1), np.diag(A2))

    def test_utils_matrix_inv(self):
        """Test utils. matrix_inv_cholesky"""

        X = np.random.randn(10000, 1000)
        A = X.T@X
        O, __ = _numpy_linalg_utils.matrix_inv_cholesky(A)
        np.testing.assert_almost_equal(O,
                                       np.linalg.inv(A))

    def test_utils_matern_type_kernel(self):
        """Test utils. matern_type_kernel"""

        h = 1.
        mat = _numpy_linalg_utils.matern_type_kernel(100, h)
        self.assertEqual(mat.shape[0], 100)
        self.assertEqual(mat.shape[1], 100)
        self.assertGreater(np.min(mat), 0.)

    def test_utils_one_hot_encoding(self):
        """Test utils. one_hot_encoding"""

        mat = _numpy_linalg_utils.one_hot_encoding(np.arange(10))
        np.testing.assert_array_equal(mat, np.eye(10))

        mat = _numpy_linalg_utils.one_hot_encoding(np.array([0, 0, 0, 1, 2]))
        self.assertEqual(mat.shape[0], 5)
        self.assertEqual(mat.shape[1], 3)
        np.testing.assert_array_equal(mat.sum(axis=0), np.array([3., 1., 1.]))

    def test_utils_prepare_smoothness_cov(self):
        """Test utils. prepare_smoothness_cov"""

        F = [np.random.randn(100, 100),
             np.random.randn(100, 10),
             np.random.randn(100, 5)]
        h = [None, None, None]
        Omega, Omega_inv = _numpy_model_utils.prepare_smoothness_cov(F, h)
        np.testing.assert_array_equal(Omega, np.eye(115))
        np.testing.assert_array_equal(Omega_inv, np.eye(115))

        F = [np.random.randn(100, 100),
             np.random.randn(100, 10),
             np.random.randn(100, 5)]
        h = [10., None, None]
        Omega, Omega_inv = _numpy_model_utils.prepare_smoothness_cov(F, h)
        mat = _numpy_linalg_utils.matern_type_kernel(100, 10.)

        np.testing.assert_array_equal(Omega[:100, :][:, :100], mat)
        np.testing.assert_array_equal(Omega[100:, :][:, 100:], np.eye(15))

    def test_utils_get_hyperparams_from_tuple(self):
        """Test utils. get_hyperparams_from_tuple"""

        tb = (1e-5, 1e-3, 1e-3, 1e-4)
        eta, tau, phi, kappa = (
            _numpy_model_utils.get_hyperparams_from_tuple(tb)
        )

        self.assertEqual(eta, 1e-5)
        self.assertEqual(tau, 1e-3)
        self.assertEqual(phi, 1e-3)
        self.assertEqual(kappa, 1e-4)

    def test_utils_compute_covariance(self):
        """Test utils. compute_covariance"""
        nu = 0.3
        F = [np.random.randn(100, 100),
             np.random.randn(100, 10),
             np.random.randn(100, 5)]
        h = [None, None, None]
        X = np.concatenate(F, axis=1)
        covX = X.T@X
        ind = np.concatenate([np.ones(100)*0, np.ones(10), np.ones(5)*2])
        mat_indexer = _numpy_linalg_utils.one_hot_encoding(ind)
        lambdas = np.array([100., 1., 5.])
        lambdas_diag = mat_indexer @ lambdas

        """ Compare without smoothness term """
        Sigma, __ = _numpy_model_utils.compute_covariance(
            nu, covX, lambdas_diag)
        dummy = np.concatenate([np.ones(100)*100., np.ones(10), np.ones(5)*5.])
        ref = np.linalg.inv(1/nu*covX+np.diag(1./dummy))
        np.testing.assert_almost_equal(Sigma, ref)

        """ Compare with smoothness term """
        h = [10., None, None]
        Omega_inv = _numpy_model_utils.prepare_smoothness_cov(F, h)[1]
        Sigma, __ = _numpy_model_utils.compute_covariance(nu, covX, lambdas_diag,
                                                          Omega_inv)
        ref = np.linalg.inv(1/nu*covX+np.diag(1./dummy)@Omega_inv)
        np.testing.assert_almost_equal(Sigma, ref)

    def test_multidim(self):
        """Compare with reference estimates."""
        def comparisons(num_obs, num_dim, P):
            """ Compare with the following:
            rand('twister', 1337);
            X = norminv(rand(num_obs,num_dim),0,1);
            Y = X(:,1) + norminv(rand(num_obs,P),0,1);
            F = {X(:,1:4), X(:,5:end)}; 
            """
            from scipy.stats import norm
            np.random.seed(1337)
            X = norm.ppf(np.random.random((num_dim, num_obs)).T)
            N = norm.ppf(np.random.random((P, num_obs)).T)
            Y = X[:, [0]] + N
            F = [X[:, :4], X[:, 4:]]
            return F, Y

        F, Y = comparisons(128, 8, 1)

        for multi_dim in [True, False]:
            for smooth in [None, [0.0001, 0.0001]]:
                emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                               max_iterations=200)
                emb.set_multidimensional(multi_dim)
                if smooth:
                    emb.set_smoothness_param(smooth)
                emb.set_compute_score(True)
                emb.fit(F, Y)

                reference = dict()
                reference['lambdas'] = [0.124752421156323, 0.000099481704179]
                reference['nu'] = 1.016172147103439
                reference['score'] = -62.375380659874189

                for key in ['lambdas', 'nu', 'score']:
                    np.testing.assert_almost_equal(
                        reference[key], emb.summary[key][-1])

        F, Y = comparisons(128, 256, 1)

        for multi_dim in [True, False]:
            for smooth in [None, [0.0001, 0.0001]]:
                emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                               max_iterations=200)
                emb.set_multidimensional(multi_dim)
                if smooth:
                    emb.set_smoothness_param(smooth)
                emb.set_compute_score(True)
                emb.fit(F, Y)

                reference = dict()
                reference['lambdas'] = [0.170212040035268, 0.000122737436378]
                reference['nu'] = 0.828597806038281
                reference['score'] = -52.269084551054725

                for key in ['lambdas', 'nu', 'score']:
                    np.testing.assert_almost_equal(
                        reference[key], emb.summary[key][-1])

        F, Y = comparisons(128, 256, 1)

        for multi_dim in [True, False]:
            for smooth in [None, [0.0001, 0.0001]]:
                emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                               max_iterations=200)
                emb.set_multidimensional(multi_dim)
                if smooth:
                    emb.set_smoothness_param(smooth)
                emb.set_compute_score(True)
                emb.fit(F, Y)

                reference = dict()
                reference['lambdas'] = [0.170212040035268, 0.000122737436378]
                reference['nu'] = 0.828597806038281
                reference['score'] = -52.269084551054725

                for key in ['lambdas', 'nu', 'score']:
                    np.testing.assert_almost_equal(
                        reference[key], emb.summary[key][-1])

        F, Y = comparisons(128, 8, 1000)

        for smooth in [None, [0.0001, 0.0001]]:
            emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                           max_iterations=200)
            emb.set_multidimensional(True)
            if smooth:
                emb.set_smoothness_param(smooth)
            emb.set_compute_score(True)
            emb.fit(F, Y)

            reference = dict()
            reference['lambdas'] = [0.250738415633565, 0.000385719557023]
            reference['nu'] = 0.994101490100018
            reference['score'] = -7.068875711264722e+04

            for key in ['lambdas', 'nu', 'score']:
                np.testing.assert_almost_equal(
                    reference[key], emb.summary[key][-1])

        F, Y = comparisons(128,8,1)
        for multi_dim in [True, False]:
            for smooth in [None, [0.0001, 0.0001]]:
                # Increase the gamma parameter in this case, and use early 
                # stopping criterion.
                emb = EMBanded(hyper_params=(1e-3, 1e-3, 1e-3, 1e-3),
                               max_iterations=200)
                emb.set_multidimensional(multi_dim)
                emb.set_early_stopping_tol(1e-8)
                if smooth:
                    emb.set_smoothness_param(smooth)
                emb.set_compute_score(True)
                emb.fit(F,Y)
                
                # This should stop after 27 iterations 
                np.testing.assert_equal(27-1, len(emb.summary['score']))
                reference = dict()
                reference['lambdas'] = [0.125100997994117,0.000948537402283]
                reference['nu'] =  1.014388456747040
                reference['score'] = -64.725400737218351
                
                for key in ['lambdas','nu','score']:
                    np.testing.assert_almost_equal(reference[key],emb.summary[key][-1])


def _compare_models(F, y):
    """Fits four models that should yield highly similar results"""
    # Model 1: fastest
    emb1 = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                    max_iterations=200)
    emb1.set_multidimensional(False)
    emb1.set_store_covariance_terms(True)
    emb1.set_compute_score(True)
    emb1.fit(F, y)

    # Model 2: slower
    emb2 = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                    max_iterations=200)
    emb2.set_multidimensional(True)
    emb2.set_store_covariance_terms(True)
    emb2.fit(F, y)

    # Model 3: include Omega ~ np.eye()
    emb3 = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                    max_iterations=200)
    emb3.set_smoothness_param([0.0001]*len(F))
    emb3.set_store_covariance_terms(True)
    emb3.fit(F, y)

    # Model 4: include Omega ~ np.eye()
    emb4 = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                    max_iterations=200)
    emb4.set_multidimensional(True)
    emb4.set_smoothness_param([0.0001]*len(F))
    emb4.set_store_covariance_terms(True)
    emb4.fit(F, y)

    np.testing.assert_array_almost_equal(emb1.W, emb2.W)
    np.testing.assert_array_almost_equal(emb1.W, emb3.W)
    np.testing.assert_array_almost_equal(emb1.W, emb4.W)
    np.testing.assert_array_almost_equal(emb2.W, emb3.W)
    np.testing.assert_array_almost_equal(emb2.W, emb4.W)
    np.testing.assert_array_almost_equal(emb3.W, emb4.W)

    np.testing.assert_array_almost_equal(
        emb1.summary['lambdas'][-1, :], emb2.summary['lambdas'][-1, :])
    np.testing.assert_array_almost_equal(
        emb1.summary['lambdas'][-1, :], emb3.summary['lambdas'][-1, :])
    np.testing.assert_array_almost_equal(
        emb1.summary['lambdas'][-1, :], emb4.summary['lambdas'][-1, :])
    np.testing.assert_array_almost_equal(
        emb2.summary['lambdas'][-1, :], emb3.summary['lambdas'][-1, :])
    np.testing.assert_array_almost_equal(
        emb2.summary['lambdas'][-1, :], emb4.summary['lambdas'][-1, :])
    np.testing.assert_array_almost_equal(
        emb3.summary['lambdas'][-1, :], emb4.summary['lambdas'][-1, :])

    num_dim = np.concatenate(F, axis=1).shape[1]
    for key in ['Omega', 'Omega_inv']:
        assert emb1.covariance_terms[key] is None
        assert emb2.covariance_terms[key] is None
        assert isinstance(emb3.covariance_terms[key], np.ndarray)
        assert isinstance(emb4.covariance_terms[key], np.ndarray)
        np.testing.assert_array_almost_equal(
            emb3.covariance_terms[key], np.eye(num_dim))
        np.testing.assert_array_almost_equal(
            emb4.covariance_terms[key], np.eye(num_dim))


if __name__ == '__main__':
    absltest.main()
