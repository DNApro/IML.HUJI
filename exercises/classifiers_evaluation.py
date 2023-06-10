from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)



def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    dir_path = "../datasets/"
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(dir_path+f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        add_to_losses = lambda p, X_dummy, y_dummy: losses.append(p._loss(X, y))
        perc = Perceptron(callback=add_to_losses)
        perc.fit(X, y)

        fig = go.Figure([go.Scatter(x=list(range(len(losses))), y=losses, mode='lines')],
                        layout=go.Layout(title="Loss as Function of Number of Iterations- " + n + " Dataset",
                                         xaxis={"title": "Number of Iterations"},
                                         yaxis={"title": "Misclassification Error"}))
        fig.write_image(f"perceptron_fit_{n}.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="DarkGray")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    dir_path = "../datasets/"
    for n, f in enumerate(["gaussian1.npy", "gaussian2.npy"]):
        # Load dataset
        X, y = load_dataset(dir_path+f)

        # Fit models and predict over training set
        lda_model, lda_name_string = LDA(), "Linear Discriminant Analysis"
        bayes_model, bayes_name_string = GaussianNaiveBayes(), "Gaussian Naive Bayes"
        lda_model.fit(X, y)
        bayes_model.fit(X, y)

        predict_lda = lda_model.predict(X)
        predict_bayes = bayes_model.predict(X)

        # Plot a subplotsure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        acc_bayes = accuracy(y, predict_bayes)
        acc_lda = accuracy(y, predict_lda)
        subplots = make_subplots(rows=1, cols=2,
                                 subplot_titles=(f"Model: {lda_name_string}, Accuracy: {round(acc_lda*100, 3)}",
                                                 f"Model: {bayes_name_string}, Accuracy: {round(acc_bayes*100, 3)}"))

        # Add traces for data-points setting symbols and colors
        subplots.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                        marker=dict(color=predict_bayes, symbol=class_symbols[y])),
                             go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                        marker=dict(color=predict_lda, symbol=class_symbols[y]))],
                            rows=[1, 1], cols=[2, 1])

        # Add `X` dots specifying fitted Gaussians' means
        subplots.add_traces([go.Scatter(x=bayes_model.mu_[:, 0], y=bayes_model.mu_[:, 1], mode="markers",
                                        marker=dict(symbol="x", color="DarkGray", size=15)),
                             go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1], mode="markers",
                                        marker=dict(symbol="x", color="DarkGray", size=15))],
                            rows=[1, 1], cols=[2, 1])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(lda_model.classes_.size):
            subplots.add_traces([get_ellipse(bayes_model.mu_[i], np.diag(bayes_model.vars_[i])),
                                 get_ellipse(lda_model.mu_[i], lda_model.cov_)],
                                rows=[1, 1], cols=[2, 1])

        subplots.update_yaxes(scaleanchor="x", scaleratio=1)
        subplots.update_layout(title_text=rf"$\text{{Comparing Gaussian Classifiers - {f[:-4]} dataset}}$",
                          width=1200, height=650, showlegend=False)

        # subplots.show()
        subplots.write_image(f"gaussian_compare_{n}.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
