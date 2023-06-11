from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error as mse
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, test_X = X[:n_samples], X[n_samples:]
    train_y, test_y = y[:n_samples], y[n_samples]


    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_lams = np.linspace(0, 10, n_evaluations)
    lasso_alps = np.linspace(0, 2.75, n_evaluations)
    ridge_train_err = np.zeros(n_evaluations)
    ridge_valid_err = np.zeros(n_evaluations)
    lasso_train_err = np.zeros(n_evaluations)
    lasso_valid_err = np.zeros(n_evaluations)

    # Lasso
    for i, lambda_val in enumerate(lasso_alps):
        lasso_model = Lasso(lambda_val, fit_intercept=True)
        lasso_train_err[i], lasso_valid_err[i] = cross_validate(lasso_model, train_X, train_y, mse)

    # Ridge
    for i, lambda_val in enumerate(ridge_lams):
        ridge_model = RidgeRegression(lambda_val)
        ridge_train_err[i], ridge_valid_err[i] = cross_validate(ridge_model, train_X, train_y, mse)

    ridge_vs_lasso_graph = make_subplots(rows=1, cols=2, subplot_titles=['Ridge Regression', 'Lasso Regression'],
                                         shared_xaxes=True)
    ridge_vs_lasso_graph.update_layout(title='Train and Validation Errors as a Function of Regularization Parameter',
                                       width=1000, height=400)

    ridge_vs_lasso_graph.add_trace(go.Scatter(x=ridge_lams, y=ridge_train_err, name='Ridge Train Error'),
                                   row=1, col=1)
    ridge_vs_lasso_graph.add_trace(go.Scatter(x=ridge_lams, y=ridge_valid_err, name='Ridge Validation Error'),
                                   row=1, col=1)
    ridge_vs_lasso_graph.add_trace(go.Scatter(x=lasso_alps, y=lasso_train_err, name='Lasso Train Error'),
                                   row=1, col=2)
    ridge_vs_lasso_graph.add_trace(go.Scatter(x=lasso_alps, y=lasso_valid_err, name='Lasso Validation Error'),
                                   row=1, col=2)

    ridge_vs_lasso_graph.write_image('Ridge_Lasso_1.png')


    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    opt_ridge_lam = ridge_lams[np.argmin(ridge_valid_err)]
    opt_lasso_alp = lasso_alps[np.argmin(lasso_valid_err)]
    opt_ridge_model = RidgeRegression(lam=opt_ridge_lam)
    opt_lasso_model = Lasso(alpha=opt_lasso_alp)
    least_squares_model = LinearRegression()
    opt_ridge_model.fit(train_X, train_y)
    opt_lasso_model.fit(train_X, train_y)
    least_squares_model.fit(train_X, train_y)
    print(f'Best Ridge lambda: {opt_ridge_lam}')
    print(f'Best Lasso alpha: {opt_lasso_alp}')
    print(f'Least Squares error: {least_squares_model.loss(test_X, test_y)}')
    print(f'Ridge error: {opt_ridge_model.loss(test_X, test_y)}')
    print(f'Lasso error: {mse(y_true=test_y, y_pred=opt_lasso_model.predict(test_X))}')


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()