import matplotlib.pyplot as plt
import numpy as np
from regression import util

from regression.linear_model import LinearModel


def main(tau, train_path, eval_path):
    """ Locally weighted regression (LWR)

    :param tau:  Bandwidth parameter for LWR
    :param train_path: Path to CSV file containing dataset for training.
    :param eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    print(mse)
    # Plot validation prediction on top of training set
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("data/ds5_b.png")
    # Plot data


class LocallyWeightedLinearRegression(LinearModel):
    """ Locally Weighted Regression(LWR)

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """ Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y

    def predict(self, x):
        """ Make predictions given inputs x.

        :param x: Inputs of shape (m, n)
        :return: Outputs of shape (m, )
        """
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            W = np.diag(np.exp(-np.sum((self.x - x[i]) ** 2, axis=1) / (2 * self.tau**2)))
            y_pred[i] = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y).T.dot(x[i])

        return y_pred


if __name__ == '__main__':
    main(0.5, "data/ds5_train.csv", "data/ds5_test.csv")
