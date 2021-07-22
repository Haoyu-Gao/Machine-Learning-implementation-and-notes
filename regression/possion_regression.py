import numpy as np
import matplotlib.pyplot as plt
from regression import util

from regression.linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """ Poisson regression with gradient ascent

    :param lr: Learning rate for gradient ascent
    :param train_path: Path to CSV file containing dataset for training
    :param eval_path: Path to CSV file containing dataset for evaluation
    :param pred_path: Path to save predictions
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    model = PoissonRegression(step_size=lr, eps=1e-5)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_eval)

    np.savetxt(pred_path, y_pred)

    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('data/ds4_p03.png')


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)

    """

    def fit(self, x, y):
        """ Run gradient ascent to maximize likelihood for Poisson regression

        :param x: Training example inputs. Shape (m, n).
        :param y: Training example labels. Shape (m, ).
        """
        m, n = x.shape
        self.theta = np.zeros(n, dtype=np.float64)

        while True:
            theta = np.copy(self.theta)

            self.theta += (1 / m) * self.step_size * x.T.dot(y - np.exp(x.dot(self.theta)))

            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break

    def predict(self, x):
        """ Make a prediction given inputs x

        :param x: Inputs of shape (m, n).
        :return: Floating-point prediction for each input, shape (m, ).
        """
        return np.exp(x.dot(self.theta))


if __name__ == '__main__':
    main(1e-7, 'data/ds4_train.csv', 'data/ds4_valid.csv', 'data/ds4_pred_p03.csv')
    # in this exercise, label y is extremely large, so we have to set learning rate small enough in order to prevent
    # parameter theta from overflow
