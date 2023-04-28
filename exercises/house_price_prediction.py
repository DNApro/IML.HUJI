from utils import split_train_test
from linear_regression import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"



def clean_up(X, y):
    X.drop_duplicates()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col])

    X["is_renovated"] = X["yr_renovated"].apply(lambda v: 0 if np.isnan(v) or int(v) == 0 else 1 if int(v) < 1990 else 2 \
        if int(v) < 2010 else 3)
    X.drop(['yr_renovated'], axis=1, inplace=True)


    X.zipcode = X.zipcode.astype(int)
    boundaries = {'bedrooms': range(1, 11), 'bathrooms': range(0, 8), 'sqft_living': range(100, 322910),
                  'sqft_lot': range(100, 322910), 'floors': range(1, 5), 'waterfront': [0, 1], 'view': range(5),
                  'condition': range(1,6), 'grade': range(1, 14), 'sqft_above': range(322910),
                  'sqft_basement': range(322910),
                  'yr_built': range(1, 2023)
                  }
    feature_validator = lambda f: X[f].isin(boundaries[f])

    for f in boundaries.keys():
        X = X[feature_validator(f)]

    y = y[X.index]
    y = y[y>1000]

    global zip_cols
    zip_cols = X["zipcode"].unique()
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

    return X, y

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    X = X.replace(['NA', 'N/A', None, np.nan], np.nan)
    for feature in X.columns:
        if feature.startswith("zipcode"):
            X[feature].fillna(0, inplace=True)

    X, y = clean_up(X, y)

    global mean_features
    mean_features = pd.Series(X.mean().to_dict())

    global train_columns
    train_columns = X.columns



    X.dropna()

    return X, y


def preprocess_test(X: pd.DataFrame):

    X.loc[~X["zipcode"].isin(zip_cols), :] = 0  # necessary to change the type of col to int
    X["zipcode"] = X["zipcode"].astype(int)
    X = pd.get_dummies(X, columns=['zipcode'])
    X = X.reindex(columns=train_columns, fill_value=0)
    for feature in X.columns:
        if feature.startswith("zipcode"):
            X[feature].fillna(0, inplace=True)
        else:
            X[feature].fillna(mean_features[feature], inplace=True)

    return X

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
    X.drop(['id', 'date', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], axis=1, inplace=True)
    X = X.replace(['NA', 'N/A', None, np.nan], np.nan)
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

    X = X.loc[:, ~(X.columns.str.startswith('zipcode'))]
    for feature in X:
        pearson = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", trendline="ols",
                         color_discrete_sequence=["black"],
                         title=f"Correlation Between {feature} Values and Response: {pearson}",
                         labels={"x": f"{feature} Values", "y": "Response Values"})
        fig.write_image(output_path + f"/pearson_correlation_{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("./datasets/house_prices.csv")

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
    percentages = range(10, 101)
    results = np.zeros((len(percentages), 10))
    for i, perc in enumerate(percentages):
        for sample in range(results.shape[1]):
            curr_tr_X = train_X.sample(frac=perc / 100.0)
            curr_tr_y = train_y.loc[curr_tr_X.index]
            model = LinearRegression(include_intercept=True).fit(curr_tr_X, curr_tr_y)
            results[i, sample] = model.loss(test_X, test_y)

    avg_of_res = results.mean(axis=1)
    std_of_res = results.std(axis=1)
    q4_fig = go.Figure([go.Scatter(x=list(percentages), y=avg_of_res - 2 * std_of_res, fill=None, mode="lines",
                                   line=dict(color="lightgrey")),
                        go.Scatter(x=list(percentages), y=avg_of_res + 2 * std_of_res, fill='tonexty', mode="lines",
                                   line=dict(color="lightgrey")),
                        go.Scatter(x=list(percentages), y=avg_of_res, mode="markers+lines", marker=dict(color="black"))],
                       layout=go.Layout(title="Mean loss tests over different training set sizes",
                                        xaxis=dict(title="Percentage of taken samples from the Training Set"),
                                        yaxis=dict(title="MSE"), showlegend=False))
    q4_fig.write_image("/mean_loss_percentage.png")
