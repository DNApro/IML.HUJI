import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def question_1_func(n_learners, noise, test_errors, train_errors):
    fig = go.Figure(
        data=[
            go.Scatter(x=list(range(1, n_learners + 1)), y=train_errors, name="Train Error", mode="lines"),
            go.Scatter(x=list(range(1, n_learners + 1)), y=test_errors, name="Test Error", mode="lines")
        ],
        layout=go.Layout(
            width=800, height=600,
            title={"x": 0.5, "text": r"$\text{AdaBoost Misclassification As Function Of Number Of Classifiers}$"},
            xaxis_title=r"$\text{Number of Fitted Learners}$",
            yaxis_title=r"$\text{Misclassification Loss}$"
        )
    )
    fig.write_image(f"adaboost_w_{noise}_noise.png")


def question_2_func(adab_model, T, lims, noise, test_X, test_y):
    fig = make_subplots(rows=1, cols=4, subplot_titles=[rf"$\text{{{t} Classifiers}}$" for t in T])
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda X: adab_model.partial_predict(X, t), lims[0], lims[1], density=60, showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x")))],
            rows=1, cols=i + 1)
    fig.update_layout(height=500, width=2000).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"adaboost_{noise}_decision_boundaries.png")


def question_3_func(adab_model, lims, noise, test_X, test_errors, test_y):
    best_t = np.argmin(test_errors) + 1
    fig = go.Figure([
        decision_surface(lambda X: adab_model.partial_predict(X, best_t), lims[0], lims[1], density=60, showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x")))],
        layout=go.Layout(width=500, height=500, xaxis=dict(visible=False), yaxis=dict(visible=False),
                         title=f"Best Performing Ensemble<br>Size: {best_t}, Accuracy: {1 - round(test_errors[best_t - 1], 2)}"))
    fig.write_image(f"adaboost_{noise}_best_over_test.png")


def question_4_func(adab_model, lims, noise, train_X, train_y):
    D = 20 * adab_model.D_ / adab_model.D_.max()
    fig = go.Figure([
        decision_surface(adab_model.predict, lims[0], lims[1], density=60, showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(size=D, color=train_y, symbol=np.where(train_y == 1, "circle", "x")))],
        layout=go.Layout(width=500, height=500, xaxis=dict(visible=False), yaxis=dict(visible=False),
                         title=f"Final AdaBoost Sample Distribution"))
    fig.write_image(f"adaboost_{noise}_weighted_samples.png")

def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    adab = AdaBoost(DecisionStump, n_learners)
    adab.fit(train_X, train_y)

    train_errors = []
    test_errors = []

    for t in range(1, n_learners + 1):
        train_loss = adab.partial_loss(train_X, train_y, t)
        test_loss = adab.partial_loss(test_X, test_y, t)
        train_errors.append(train_loss)
        test_errors.append(test_loss)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    question_1_func(n_learners, noise, test_errors, train_errors)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    question_2_func(adab, T, lims, noise, test_X, test_y)

    # Question 3: Decision surface of best performing ensemble
    question_3_func(adab, lims, noise, test_X, test_errors, test_y)

    # Question 4: Decision surface with weighted samples
    question_4_func(adab, lims, noise, train_X, train_y)


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise)
