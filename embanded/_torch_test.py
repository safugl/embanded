# pylint: skip-file

import copy
from absl.testing import absltest
import numpy as np
import torch
import pytest
from .embanded_torch import EMBanded


class EMBandedTests(absltest.TestCase):
    """Test cases for EMBanded."""

    def test_compare_model_settings(self):
        """Fit models with different settings and compare outputs"""

        torch.manual_seed(0)
        num_obs = 400
        for num_groups in [1, 2, 4, 8, 16, 32]:
            F = [[]]*num_groups
            W = [[]]*num_groups
            for f in range(num_groups):
                F[f] = torch.randn(num_obs, 5)
                if torch.rand(1) < 0.5:
                    W[f] = torch.randn(5, 1)
                else:
                    W[f] = torch.zeros((5, 1))

            X = torch.concatenate(F, axis=1)
            W = torch.concatenate(W, axis=0)
            N = torch.randn(num_obs, 1)*2
            Y = X@W + N
            X_before = copy.deepcopy(X)
            Y_before = copy.deepcopy(Y)
            _compare_models(F, Y)

            np.testing.assert_equal(X.numpy(), X_before.numpy())
            np.testing.assert_equal(Y.numpy(), Y_before.numpy())

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

    def test_fails_input_data(self):
        """Tests cases for wrong input data, F and y."""

        with pytest.raises(Exception):
            # F should be a list of tensors
            emb = EMBanded()
            emb.fit(torch.randn(100, 10), torch.randn(100, 1))

        with pytest.raises(Exception):
            # y has too many rows.
            emb = EMBanded()
            F = [torch.randn(100, 1), torch.randn(100, 10),
                 torch.randn(100, 10)]
            emb.fit(F, np.random.randn(1000, 1))

        with pytest.raises(Exception):
            # y has too many columns.
            emb = EMBanded()
            F = [torch.randn(100, 1), torch.randn(100, 10),
                 torch.randn(100, 10)]
            emb.fit(F, np.random.randn(100, 10))

    def test_fails_initializations(self):
        """Tests cases for wrong hyper parameter intializations"""

        with pytest.raises(Exception):
            emb = EMBanded()
            emb.set_lambdas_init(1.)

        with pytest.raises(Exception):
            torch.manual_seed(1)
            emb = EMBanded()
            emb.set_lambdas_init(torch.randn(100))

        with pytest.raises(Exception):
            emb = EMBanded()
            emb.set_nu_init(None)

        with pytest.raises(Exception):
            emb = EMBanded()
            emb.set_multidimensional(None)

    def test_multidim(self):
        """Test multidimensional model."""
        def comparisons(num_obs, num_dim, P):
            """ Compare with the following:
            rand('twister', 1337);
            X = norminv(rand(num_obs,num_dim),0,1);
            Y = X(:,1) + norminv(rand(num_obs,P),0,1);
            F = {X(:,1:4), X(:,5:end)}; 
            """
            from scipy.stats import norm
            np.random.seed(1337)
            X = torch.from_numpy(norm.ppf(np.random.random((num_dim,num_obs)).T))
            N = torch.from_numpy(norm.ppf(np.random.random((P,num_obs)).T))
            Y = X[:,[0]] +  N
            F = [X[:,:4], X[:,4:]]
            return F, Y

        F, Y = comparisons(128,8,1)

        for multi_dim in [True, False]:
            for smooth in [None, [0.0001, 0.0001]]:
                emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                               max_iterations=200)
                emb.set_multidimensional(multi_dim)
                if smooth:
                    emb.set_smoothness_param(smooth)
                emb.set_compute_score(True)
                emb.fit(F,Y)
                
                reference = dict()
                reference['lambdas'] = [0.124752421156323, 0.000099481704179]
                reference['nu'] = 1.016172147103439
                reference['score'] = -62.375380659874189
                
                for key in ['lambdas','nu','score']:
                    np.testing.assert_almost_equal(reference[key],emb.summary[key][-1])
                

        F, Y = comparisons(128,256,1)

        for multi_dim in [True, False]:
            for smooth in [None, [0.0001, 0.0001]]:
                emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                               max_iterations=200)
                emb.set_multidimensional(multi_dim)
                if smooth:
                    emb.set_smoothness_param(smooth)
                emb.set_compute_score(True)
                emb.fit(F,Y)
                
                reference = dict()
                reference['lambdas'] = [0.170212040035268, 0.000122737436378]
                reference['nu'] =  0.828597806038281
                reference['score'] = -52.269084551054725
                
                for key in ['lambdas','nu','score']:
                    np.testing.assert_almost_equal(reference[key],emb.summary[key][-1].numpy())
                

        F, Y = comparisons(128,8,1000)

        for smooth in [None, [0.0001, 0.0001]]:
            emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                           max_iterations=200)
            emb.set_multidimensional(True)
            if smooth:
                emb.set_smoothness_param(smooth)
            emb.set_compute_score(True)
            emb.fit(F,Y)
            
            reference = dict()
            reference['lambdas'] = [0.250738415633565,0.000385719557023]
            reference['nu'] =  0.994101490100018
            reference['score'] =  -7.068875711264722e+04
            
            for key in ['lambdas','nu','score']:
                np.testing.assert_almost_equal(reference[key],emb.summary[key][-1].numpy())
                
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
    """Fits four models that should yield highly similar results."""
    # Model 1: fastest
    emb1 = EMBanded(
        hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
        max_iterations=200)
    emb1.set_multidimensional(False)
    emb1.set_store_covariance_terms(True)
    emb1.fit(F, y)

    # Model 2: slower
    emb2 = EMBanded(
        hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
        max_iterations=200)
    emb2.set_multidimensional(True)
    emb2.set_store_covariance_terms(True)
    emb2.fit(F, y)

    # Model 3: include Omega ~ np.eye()
    emb3 = EMBanded(
        hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
        max_iterations=200)
    emb3.set_smoothness_param([0.0001]*len(F))
    emb3.set_store_covariance_terms(True)
    emb3.fit(F, y)

    # Model 4: include Omega ~ np.eye()
    emb4 = EMBanded(
        hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
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
        assert isinstance(emb3.covariance_terms[key], torch.Tensor)
        assert isinstance(emb4.covariance_terms[key], torch.Tensor)
        np.testing.assert_array_almost_equal(
            emb3.covariance_terms[key], torch.eye(num_dim))
        np.testing.assert_array_almost_equal(
            emb4.covariance_terms[key], torch.eye(num_dim))


if __name__ == '__main__':
    absltest.main()
