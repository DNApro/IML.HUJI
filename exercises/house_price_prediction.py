from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from anaconda3.Lib.typing import Optional

pio.templates.default = "simple_white"

global MEAN_FEATURES
global MEAN_RESPONSES

boundaries = {'bedrooms': range(1, 7), 'bathrooms': range(1, 8), 'sqft_living': range(500, 18837),
    'sqft_lot': range(500, 18837), 'floors': range(1, 5), 'waterfront': [0, 1],  'view': range(5),
    'condition': range(6), 'grade': range(1, 14), 'sqft_above': [0, np.inf], 'sqft_basement': [0, np.inf]
}
feature_validator = lambda f : X[f].isin(boundaries[f])

def clean_up(X, y):
    X.drop(['date', 'lat', 'long', 'id'], axis=1, inplace=True)
    mask = X.bedrooms.isin(range(1,7)) and X.bathrooms.isin(range(1,8)) and X.sqft_living.isin(range(500, 18837)) \
           and X.sqft_lot.isin(range(500, 18837)) and X.floors.isin(range(1, 5)) and X.waterfront.isin([0,1]) \
           and X.view.isin(range(5)) and X.condition.isin(range(6)) and X.grade.isin(range(1,14)) and \
            X.sqft_above >= 0 and X.sqft_basement >= 0
    X.zipcode = X.zipcode.astype(int)
    X = X.get_dummies(X, columns=['zipcode'])
    X = X[mask]
    y = y[mask]
    return X, y

def preprocess_train(X: pd.DataFrame, y: pd.Series):


    X,y = clean_up(X, y)

    MEAN_FEATURES = np.nanmean(X)
    #TODO do it after clean up
    for feature in X.columns:
        X[feature].fillna(np.nanmean(X[feature]), inplace=True)


    return X,y


def preprocess_test(X: pd.DataFrame):
    pass

def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if y is not None:
        return preprocess_train(X, y)
    return preprocess_test(X)




def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sigma_y = np.std(y)
    X = pd.get_dummies(X)

    fig = go.Figure()
    for feature in X.columns:
        col = X[feature]
        pearson = np.cov(col, y)/(np.std(col)*sigma_y)

        fig.add_trace(go.Scatter(x=col, y=y, mode="markers+lines"))
        fig.update_layout(title=f"Corellation between {feature} and Response values.   P.C = {pearson}",
                          xaxis_title=f"{feature}", yaxis_title="response", height=400, width=1200)
    fig.write_image(output_path + "/images/fig.png")



if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    df = df[~np.isnan(df.price)]
    X = df.drop('price', axis=1)
    y = df['price']
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
