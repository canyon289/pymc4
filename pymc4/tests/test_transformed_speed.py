"""
Placeholder test for RV Instantiation and run
"""
from ..random_variables import Normal


def test_speed(tf_session):
    dist = Normal(name="test_dist", mu=0, sigma=1)

    log_probs = []
    for _ in range(1000):
        log_prob_value = dist.log_prob()
        log_probs.append(log_prob_value)
    tf_session.run([log_probs])


def test_with_context(tf_session):
    """
    Just a sketch, probably won't work.

    The intent here is to provide a contrasting example to the above 
    test (`test_speed`), in which here, we instantiate the InferenceContext.

    This lets us measure speed of code both with and without InferenceContext,
    amongst multiple implementations.

    Note to future self: UPDATE THIS DOCSTRING WITH FINAL CONCLUSIONS.
    """
    dist = Normal(name="normal", mu=0, sigma=1)
    dist.ctx = InferenceContext()
    dist.log_prob()

    for _ in range(1000):
        log_prob_value = dist.log_prob()
        log_probs.append(log_prob_value)

    tf_session.run([log_probs])