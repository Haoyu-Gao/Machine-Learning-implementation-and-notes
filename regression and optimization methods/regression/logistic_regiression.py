import numpy as np
import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """

    :param train_path:
    :param eval_path:
    :param pred_path:
    :return:
    """

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)


class LogisticRegression(LinearModel):
    """ Logistic regression with Newton's Method as the solver

    Example usage:
        > clf = LogisticeRegreesion()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """ Run Newton's Method to minimize J(theta) for logistic regression

        :param x: Training example inputs. Shape (m, n).
        :param y: Training example labels. Shape (m, ).
        """

    def predict(self, x):
        """ Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n)
        :return: Outputs of shape (m, ).
        """