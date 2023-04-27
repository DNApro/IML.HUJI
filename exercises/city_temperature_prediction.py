from IMLearn.learners.regressors import PolynomialFitting
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors import PolynomialFitting as PolyFit
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"



def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    X = pd.read_csv(filename, parse_dates=True)
    X = X[~np.isnan(X)]
    X["DayOfYear"] = X["Date"].dt.dayofyear
    X["Year"] = X["Year"].astype(str)
    X = X[X["Temp"] > -10]
    return X.drop_duplicates()

def question2_solution(X):
    IL_df = X[X["Country"] == 'Israel']
    fig_avg_temp_IL = px.scatter(IL_df, x="DayOfYear", y="Temp", color="Year")
    fig_avg_temp_IL.update_layout(title="The avg temp in Israel as function of the day of year",
                                  xaxis_title="Day of Year", yaxis_title="Temperature")
    fig_avg_temp_IL.write_image("./Q2_average_temp_in_israel_daily.png", engine="orca")

    fig_std_by_month_IL = px.bar(IL_df.groupby(["Month"], as_index=False).agg(['Temp', 'std']))
    fig_std_by_month_IL.update_layout(title="The std of temperature in Israel as a function of each month",
                                      xaxis_title="Month", yaxis_title="std")
    fig_std_by_month_IL.write_image("./Q2_std_of_temp_in_israel_monthly.png")

    return IL_df


def question3_solution(X):
    group_by_country_month = X.groupby(["Country", "Month"])["Temp"].agg(['mean', 'std']).reset_index()
    fig_country_month_grouped = px.line(group_by_country_month, x="Month", y="mean",
                                        color="Country", error_y="std")
    fig_country_month_grouped.update_layout(title="The avg temperature of each month grouped by countries.",
                                            xaxis_title="month", yaxis_title="temperature")
    fig_country_month_grouped.write_image("./Q3_average_temp_country_month.png")

def question4_solution(IL_df):
    train_X, train_y, test_X, test_y = split_train_test(IL_df.DayOfYear, IL_df.Temp)
    losses = []
    for k in range(1, 11):
        losses.append(round(PolyFit(k).fit(train_X, train_y).loss(test_X, test_y), 2))

    fig_poly_model_loss = px.bar(x=list(range(1, 11)), y=losses)
    fig_poly_model_loss.update_layout(title="MSE as function of k degree",
                                      xaxis_title="K", yaxis_title="MSE")
    fig_poly_model_loss.update_traces(text=[f"MSE: {l:.2f}" for l in losses])
    fig_poly_model_loss.write_image("./Q4_Israel's_MSE_as_function_of_deg.png")
    print(losses)


def question5_solution(X, IL_df):
    model = PolynomialFitting(5).fit(IL_df.DayOfYear.to_numpy(), IL_df.Temp.to_numpy())
    fig = px.bar(pd.DataFrame([{"country": country, "loss": round(model.loss(X[X["Country"] == country]["DayOfYear"],
                                                                             X[X["Country"] == country]["Temp"]), 2)}
                               for country in ["Jordan", "South Africa", "The Netherlands"]]),
                 x="country", y="loss", text="loss", color="country",
                 title="Loss Over Countries For Model Fitted Over Israel")
    fig.write_image("Test Other Countries.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    IL_df = question2_solution(X)

    # Question 3 - Exploring differences between countries
    question3_solution(X)

    # Question 4 - Fitting model for different values of `k`
    question4_solution(IL_df)

    # Question 5 - Evaluating fitted model on different countries
    question5_solution(X, IL_df)