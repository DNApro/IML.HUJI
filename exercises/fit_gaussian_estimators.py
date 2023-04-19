import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"

SAMPLES_AMOUNT = 1000
MIN_SAMPLES_AMOUNT = 10
SAMPLES_STEP = 10

UNIVAR_MU = 10
UNIVAR_SIGMA = 1

Q2_X_AXIS_NAME = "number of samples"
Q2_Y_AXIS_NAME = "|estimatedVal - trueVal|"
Q2_TITLE = "The distance of the estimated value from the true value as a function of samples amount"
Q3_X_AXIS_NAME = "Samples"
Q3_Y_AXIS_NAME = "The PDF of a sample"
Q3_TITLE = "The PDF distribution of the samples"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(UNIVAR_MU, UNIVAR_SIGMA, SAMPLES_AMOUNT)
    q1_model = UnivariateGaussian().fit(samples)
    print("(" + str(np.round(q1_model.mu_, 3)) + ", " + str(np.round(q1_model.var_, 3)) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    true_val = q1_model.mu_
    xaxis = np.arange(0, SAMPLES_AMOUNT / SAMPLES_STEP, dtype=int) + 1
    xaxis = xaxis * 10
    distances = []
    for index in xaxis:
        estimated_val = UnivariateGaussian().fit(samples[:index]).mu_
        distances.append(np.abs(estimated_val - UNIVAR_MU))
    fig_q2 = go.Figure()
    fig_q2.add_trace(go.Scatter(x=xaxis, y=distances, mode="markers+lines"))
    fig_q2.update_layout(title=Q2_TITLE, xaxis_title=Q2_X_AXIS_NAME, yaxis_title=Q2_Y_AXIS_NAME, height=400, width=1200)
    fig_q2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = q1_model.pdf(samples)
    fig_q3 = go.Figure()
    fig_q3.add_trace(go.Scatter(x=samples, y=pdf, mode="markers"))
    fig_q3.update_layout(title=Q3_TITLE, xaxis_title=Q3_X_AXIS_NAME, yaxis_title=Q3_Y_AXIS_NAME, height=400, width=1200)
    fig_q3.show()


MULTIVAR_MU = np.array([0, 0, 4, 0])
MULTIVAR_SIGMA = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
mu = lambda f1, f3: np.array([f1, 0, f3, 0])


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(MULTIVAR_MU, MULTIVAR_SIGMA, SAMPLES_AMOUNT)
    q4_model = MultivariateGaussian().fit(samples)
    print(np.round(q4_model.mu_, 3))
    print(np.round(q4_model.cov_, 3))

    # Question 5 - Likelihood evaluation
    f_values1 = np.linspace(-10, 10, 200)
    f_values3 = np.linspace(-10, 10, 200)
    multivar_log_like = np.zeros((200, 200))
    max_features = (None, None, np.NINF)

    for i, f1 in enumerate(f_values1):
        for j, f3 in enumerate(f_values3):
            curr_ll = MultivariateGaussian.log_likelihood(mu(f1, f3), MULTIVAR_SIGMA, samples)

            if curr_ll > max_features[2]:
                max_features = (f1, f3, curr_ll)

            multivar_log_like[i, j] = curr_ll

    fig = go.Figure(go.Heatmap(z=multivar_log_like))
    fig.update_layout(title='Log Likelihood Heatmap', xaxis_title='f1', yaxis_title='f3')
    fig.show()

    # Question 6 - Maximum likelihood
    print(str(np.round(max_features[0], 3)) + " " + str(np.round(max_features[1])))
    return 0


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
