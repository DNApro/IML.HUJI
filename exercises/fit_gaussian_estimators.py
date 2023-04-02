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
Q2_TITLE = "The distance of the estimated value from the true value as a function of samples amount"



def test_univariate_gaussian():
    np.random.seed(0)
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(UNIVAR_MU, UNIVAR_SIGMA, UNIVAR_SAMPLES_AMOUNT)
    q1_model = UnivariateGaussian().fit(samples)
    print("(" + str(q1_model.mu_) + ", " + str(q1_model.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    true_value = q1_model.mu_
    x_axis = np.arange(10, 1001, 10)
    y_axis = []

    for i in range(len(x_axis)):
        model = UnivariateGaussian().fit(samples[:i + 1])
        estimated = model.mu_
        distance = np.abs(estimated - true_value)
        y_axis.append(distance)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(10,1001,10)), y=y_axis, mode="markers"))
    fig.update_layout(title=Q2_TITLE, xaxis_title=Q2_X_AXIS_NAME, yaxis_title=Q2_Y_AXIS_NAME)
    fig.show()

    # mu_hat = [np.abs(true_value - UnivariateGaussian().fit(samples[:n]).mu_)
    #           for n in np.arange(1, len(samples) / 10, dtype=np.int) * 10]
    # go.Figure(go.Scatter(x=list(range(0, len(mu_hat)*10, 10)), y=mu_hat, mode="markers", marker=dict(color="black")),
    #           layout=dict(template="simple_white",
    #                       title="Deviation of Sample Mean Estimation As Function of Sample Size",
    #                       xaxis_title=r"$\text{Sample Size }n$",
    #                       yaxis_title=r"$\text{Sample Mean Estimator }\hat{\mu}_n$")) \
    #     .write_image("mean.deviation.over.sample.size.png")
    # true_value = q1_model.mu_
    # x_axis = np.linspace(10, 1010, 100)
    # y_axis = []
    #
    # for i in range(10, 1010, 10):
    #     gaus = UnivariateGaussian()
    #     gaus.fit(samples[:i])
    #     estimated = gaus.mu_
    #     distance = np.abs(estimated - true_value)
    #     y_axis.append(distance)
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="markers+lines"))
    # fig.update_layout(title="17", xaxis_title="18", yaxis_title="19", height=300)
    # fig.write_image("q2_graph_3.png")
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
    #test_multivariate_gaussian()
