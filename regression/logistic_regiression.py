import numpy as np
from regression import util
from regression.linear_model import LinearModel
import matplotlib.pyplot as plt


def main(train_path, eval_path, pred_path):
    """

    :param train_path: Path to CSV file containing dataset for training
    :param eval_path: Path to CSV file containing dataset for evaluation
    :param pred_path: Path to save predictions
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_prediction = logistic_regression_model.predict(x_eval)
    util.plot(x_eval, y_eval, logistic_regression_model.theta.reshape(3, ), "data/ds1_val_pred")
    util.plot(x_train, y_train, logistic_regression_model.theta.reshape(3, ), "data/ds1_train_pred")
    # np.savetxt(pred_path, y_prediction)

    # print the error rate
    err = 0
    for i in range(y_eval.shape[0]):
        if y_eval[i] != y_prediction[i]:
            err += 1
    print("error rate is {}".format(err / x_eval.shape[0]))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(LinearModel):
    """ Logistic regression with Newton's Method as the solver

    Example usage:
        > clf = LogisticeRegreesion()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def cost(self, h, y):
        loss = - np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=0)
        return loss / h.shape[0]

    def fit(self, x, y):
        """ Run Newton's Method to minimize J(theta) for logistic regression

        :param x: Training example inputs. Shape (m, n).
        :param y: Training example labels. Shape (m, ).
        """
        # initialization
        # y = y.reshape((x.shape[0], 1))  # (m, 1)
        self.theta = np.zeros((x.shape[1], ))   # (n+1, )
        for i in range(self.max_iter):
            # forward
            h = sigmoid(np.dot(x, self.theta))
            # backward
            gradient = 1 / x.shape[0] * (np.dot(x.T, (h - y)))
            hessian = (1 / x.shape[0]) * np.dot(x.T * h * (1 - h), x)
            # breaking condition
            theta_old = np.copy(self.theta)
            self.theta = self.theta - np.dot(np.linalg.inv(hessian), gradient)
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break

    def predict(self, x):
        """ Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n)
        :return: Outputs of shape (m, ).
        """
        y = sigmoid(np.dot(x, self.theta))
        res = np.zeros((y.shape[0], ))
        for i in range(y.shape[0]):
            res[i] = 1 if y[i] >= 0.5 else 0
        return res


# do a little test
if __name__ == "__main__":
    main("data/ds1_train.csv", "data/ds1_valid.csv", "data/ds1_output.csv")


