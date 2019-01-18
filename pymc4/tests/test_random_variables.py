"""
Tests for PyMC4 random variables
"""
import pytest


from .. import _random_variables


_TFP_SUPPORTED_ARGS = {
    _random_variables.Bernoulli: {"probs": 0.5},
    _random_variables.Beta: {"concentration0": 1, "concentration1": 1},
    # _random_variables.Binomial: {"total_count": 5.0, "probs": 0.5, "sample": 1},
    _random_variables.Categorical: {"probs": [0.1, 0.5, 0.4]},
    _random_variables.Cauchy: {"loc": 0, "scale": 1},
    _random_variables.Chi2: {"df": 2},
    _random_variables.Dirichlet: {"concentration": [1, 2], "sample": [0.5, 0.5]},
    # "Exponential": {"rate": 1},
    # "Gamma": {"concentration": 3.0, "rate": 2.0},
    # "Geometric": {"probs": 0.5, "sample": 10},
    # "Gumbel": {"loc": 0, "scale": 1},
    # "HalfCauchy": {"loc": 0, "scale": 1},
    # "HalfNormal": {"scale": 3.0},
    # "InverseGamma": {"concentration": 3, "rate": 2},
    # "InverseGaussian": {"loc": 1, "concentration": 1},
    # "Kumaraswamy": {"concentration0": 0.5, "concentration1": 0.5},
    # "LKJ": {"dimension": 1, "concentration": 1.5, "sample": [[1]]},
    # "Laplace": {"loc": 0, "scale": 1},
    # "LogNormal": {"loc": 0, "scale": 1},
    # "Logistic": {"loc": 0, "scale": 3},
    # "Multinomial": {"total_count": 4, "probs": [0.2, 0.3, 0.5], "sample": [1, 1, 2]},
    # "MultivariateNormalFullCovariance": {
    #     "loc": [1, 2],
    #     "covariance_matrix": [[0.36, 0.12], [0.12, 0.36]],
    #     "sample": [1, 2],
    # },
    # "NegativeBinomial": {"total_count": 5, "probs": 0.5, "sample": 5},
    # "Normal": {"loc": 0, "scale": 1},
    # "Pareto": {"concentration": 1, "scale": 0.1, "sample": 5},
    # "Poisson": {"rate": 2},
    # "StudentT": {"loc": 0, "scale": 1, "df": 10},
    # "Triangular": {"low": 0.0, "high": 1.0, "peak": 0.5},
    # "Uniform": {"low": 0, "high": 1},
    # "VonMises": {"loc": 0, "concentration": 1},
    # "Wishart": {"df": 3, "scale_tril": [[1]], "sample": [[1]]},
}


def test_tf_session_cleared(tf_session):
    """Temporary test: Check that fixture is finalizing correctly"""
    assert len(tf_session.graph.get_operations()) == 0


@pytest.mark.parametrize(
    "tf_distribution", _TFP_SUPPORTED_ARGS.items()
)
def test_rvs_logp_and_forward_sample(tf_session, tf_distribution):
    """Test all TFP supported distributions"""
    _dist, kwargs = tf_distribution

    sample = kwargs.pop("sample", 0.1)

    dist = _dist("test_dist", **kwargs, validate_args=True)

    if tf_distribution != "Binomial":
        # Assert that values are returned with no exceptions
        log_prob = dist.log_prob()
        vals = tf_session.run([log_prob], feed_dict={dist._backend_tensor: sample})
        assert vals is not None

    else:
        # TFP issue ticket for Binom.sample_n https://github.com/tensorflow/probability/issues/81
        assert tf_distribution == "Binomial"
        with pytest.raises(NotImplementedError) as err:
            dist.log_prob()
            assert "NotImplementedError: sample_n is not implemented: Binomial" == str(err)
