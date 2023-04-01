import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"

UNIVAR_SAMPLES_AMOUNT = 1000
UNIVAR_MU = 10
UNIVAR_SIGMA = 1

MIN_SAMPLES_AMOUNT = 10
SAMPLES_STEP = 10


Q2_X_AXIS_NAME = "number of samples"
Q2_Y_AXIS_NAME = "|estimatedVal - trueVal|"



def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    q1_samples = np.random.normal(UNIVAR_MU, UNIVAR_SIGMA, UNIVAR_SAMPLES_AMOUNT)
    q1_model = UnivariateGaussian().fit(q1_samples)
    print("(" + str(q1_model.mu_) + ", " + str(q1_model.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    distances = []
    true_val = q1_model.mu_
    # for sampSize in range(MIN_SAMPLES_AMOUNT, UNIVAR_SAMPLES_AMOUNT+1, SAMPLES_STEP):
    #     estimated_val = UnivariateGaussian().fit(q1_samples[:sampSize]).mu_
    #     distances.append(np.abs(estimated_val-true_val))

    for sampSize in np.arange(1, UNIVAR_SAMPLES_AMOUNT/10, dtype=np.int)*10:
        distances.append(np.abs(UnivariateGaussian().fit(q1_samples[:sampSize]).mu_))


    x_axis = list(range(len(distances)))
    y_axis = distances
    q2_df = pd.DataFrame({Q2_X_AXIS_NAME: x_axis, Q2_Y_AXIS_NAME: y_axis})
    fig = px.scatter(q2_df, x=Q2_X_AXIS_NAME, y=Q2_Y_AXIS_NAME, title="how far are the estimated values of different amount of samples from the true value.")
    fig.show()

    # mu_hat = [np.abs(UNIVAR_MU - UnivariateGaussian().fit(q1_samples[:n]).mu_)
    #           for n in np.arange(1, len(q1_samples) / 10, dtype=np.int) * 10]
    # go.Figure(go.Scatter(x=list(range(len(mu_hat))), y=mu_hat, mode="markers", marker=dict(color="black")),
    #           layout=dict(template="simple_white",
    #                       title="Deviation of Sample Mean Estimation As Function of Sample Size",
    #                       xaxis_title=r"$\text{Sample Size }n$",
    #                       yaxis_title=r"$\text{Sample Mean Estimator }\hat{\mu}_n$")) \
    #     .write_image("mean.deviation.over.sample.size.png")

    #
    # # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    #
    # # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    #
    # # Question 6 - Maximum likelihood
    # raise NotImplementedError()
    return 0

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
