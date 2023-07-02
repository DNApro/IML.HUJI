import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    _vals = []
    _weights = []

    def cb_func(solver: GradientDescent, weights: np.ndarray,
                val: np.ndarray, grad: np.ndarray, t: int, eta: float, delta: float):
        _vals.append(val)
        _weights.append(weights)

    return cb_func, _vals, _weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    lrs = [FixedLR(eta) for eta in etas]

    l1_vals, l2_vals = [], []


    for i in range(len(etas)):
        l1_module = L1(init)
        l2_module = L2(init)
        l1_cb, l1_rec_vals, l1_rec_weights = get_gd_state_recorder_callback()
        l2_cb, l2_rec_vals, l2_rec_weights = get_gd_state_recorder_callback()

        gd_l1 = GradientDescent(learning_rate=lrs[i], callback=l1_cb)
        w_l1 = gd_l1.fit(l1_module, None, None)
        l1_vals.append(l1_rec_vals)
        plot_l1 = plot_descent_path(L1, np.array(l1_rec_weights),
                                    title=f"L1 Descent Path With Eta = {etas[i]}")
        plot_l1.write_image(f"L1_with_eta_{etas[i]}.png")
        # plot_l1.show()

        gd_l2 = GradientDescent(learning_rate=lrs[i], callback=l2_cb)
        w_l2 = gd_l2.fit(l2_module, None, None)
        l2_vals.append(l2_rec_vals)
        plot_l2 = plot_descent_path(L2, np.array(l2_rec_weights),
                                    title=f"L2 Descent Path With Eta = {etas[i]}")
        plot_l2.write_image(f"L2_with_eta_{etas[i]}.png")
        # plot_l2.show()

    l1_iterations = np.arange(max([len(values) for values in l1_vals]))
    l1_values_fig = go.Figure()

    l2_iterations = np.arange(max([len(values) for values in l2_vals]))
    l2_values_fig = go.Figure()

    for i in range(len(etas)):
        l1_values_fig.add_trace(go.Scatter(x=l1_iterations, y=l1_vals[i],
                                           mode='markers+lines', marker=dict(size=4.3),
                                           name=f"eta = {etas[i]}"))
        l2_values_fig.add_trace(go.Scatter(x=l2_iterations, y=l2_vals[i],
                                           mode='markers+lines', marker=dict(size=4.3),
                                           name=f"eta = {etas[i]}"))

    l1_values_fig.update_layout(
        title="L1 Norms As A Function Of The GD Iteration",
        xaxis_title="Iteration", yaxis_title="L1_Norms", height=300).write_image("party1.png")
    l2_values_fig.update_layout(
        title="L2 Norms As A Function Of The GD Iteration",
        xaxis_title="Iteration", yaxis_title="L2_Norms", height=300).write_image("party2.png")

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_regress = LogisticRegression(solver=GradientDescent(FixedLR(0.0001), max_iter=20000))
    logistic_regress.fit(X_train, y_train)
    y_prob = logistic_regress.predict_proba(X_test)

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    from utils import custom
    c = [custom[0], custom[-1]]
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         marker=dict(size=5, color=c[1][1]),
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.write_image("logistic_regression_1.png")

    alpha_star = thresholds[np.argmax(tpr - fpr)]
    print(f"a* = {alpha_star}")

    logistic_regress.alpha_ = alpha_star
    print(f"The test error using a* is: {logistic_regress.loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    from IMLearn.model_selection.cross_validate import cross_validate
    from IMLearn.metrics.loss_functions import misclassification_error as MCE
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    logistic_regress.alpha_ = 0.5
    for penalty in ["l1", "l2"]:
        logistic_regress.penalty_ = penalty
        best_lam = 0
        best_validation_score = None
        for lam in lambdas:
            logistic_regress.lam_ = lam
            current_validation_score = cross_validate(logistic_regress, X_train, y_train,MCE)[1]
            if best_validation_score is None or best_validation_score > \
                    current_validation_score:
                best_validation_score = current_validation_score
                best_lam = lam
        logistic_regress.lam_ = best_lam
        logistic_regress.fit(X_train, y_train)
        test_error = logistic_regress.loss(X_test, y_test)
        print(f"{penalty} best lambda is {best_lam} and the test error is "
              f"{test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
