import numpy as np
from regression import util

from regression.linear_model import LinearModel


# the error rate from GDA is higher on dataset 1, which is understandable since from the plot, it is clear that the
# data does not obey Gaussian distribution, especially from x2 axis.
# On dataset 2, the error rate are identical as the distribution of data is closer to Gaussian distribution and our
# assumption on GDA holds. In this case, GDA model is faster to train since all the parameters only involves counting
# instead of iterative gradient descent like logistic regression.
def main(train_path, eval_path, pred_path):
    """

    :param train_path: Path to CSV file containing dataset for training
    :param eval_path: Path to CSV file containing dataset for evaluation
    :param pred_path: Path to save predictions
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    gda = GDA()
    gda.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_predict = gda.predict(x_eval)
    m = y_eval.shape[0]

    # Plot data and decision boundary
    util.plot(x_train, y_train, gda.theta, "data/ds2_pred_gda")

    # save predictions
    np.savetxt(pred_path, y_predict > 0.5, fmt="%d")

    err = 0
    for i in range(m):
        if y_eval[i] != y_predict[i]:
            err += 1
    print("error rate = " + str(err / m))


class GDA(LinearModel):
    """ Gaussian Discriminant Analysis

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """ Fit a GDA model to training set given by x and y

        :param x: Training example inputs. Shape (m, n).
        :param y: Training example labels. Shape(m, ).
        :return: theta: GDA model parameters
        """
        # positive label y = 1, negative label y = 0
        m, n = x.shape
        self.theta = np.zeros(n + 1)
        # y_format = y.reshape((y.shape[0], -1))

        # positive = 0
        # for i in range(m):
        #     positive += y[i]
        positive = np.sum(y == 1)
        phi = positive / m
        # mu0 = np.sum((1 - y_format) * x, axis=0) / (m - positive)  # (1, n)
        mu0 = np.sum(x[y == 0], axis=0) / (m - positive)
        # mu1 = np.sum(y_format * x, axis=0) / positive  # (1, n)
        mu1 = np.sum(x[y == 1], axis=0) / positive

        # diff = np.zeros(x.shape)
        # for i in range(m):
        #     diff[i] = x[i, :] - mu0 if y[i] == 0 else x[i, :] - mu1  # (m, n)
        #
        # sigma = np.zeros((n, n))
        # for i in range(m):
        #     a = diff.T[:, i].reshape((x.shape[1], -1))
        #     b = diff[i, :].reshape((-1, x.shape[1]))
        #     sigma += np.dot(a, b)
        # sigma /= m
        #
        # self.theta = (phi, mu0, mu1, sigma)
        sigma = ((x[y == 0] - mu0).T.dot(x[y == 0] - mu0) + (x[y == 1] - mu1).T.dot(x[y == 1] - mu1)) / m

        # compute theta
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = 0.5 * (mu0 + mu1).dot(sigma_inv).dot(mu0 - mu1) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv.dot(mu1 - mu0)

        return self.theta

    def predict(self, x):
        """ Make a prediction given new inputs x

        :param x: Inputs of shape (m, n).
        :return: outputs of shape (m, ).
        """
        # phi, mu0, mu1, sigma = self.theta
        #
        # mu0 = mu0.reshape((1, x.shape[1]))
        # mu1 = mu1.reshape((1, x.shape[1]))  # (1, n)
        #
        # p0 = np.zeros((x.shape[0], 1))
        # p1 = np.zeros((x.shape[0], 1))
        #
        # for i in range(x.shape[0]):
        #     p0[i] = 1 / np.power(np.linalg.det(sigma), 1 / 2) * \
        #      np.exp(-(x[i, :] - mu0).dot(np.linalg.inv(sigma).dot((x[i, :] - mu0).T)) / 2) * (1 - phi)
        #     p1[i] = 1 / np.power(np.linalg.det(sigma), 1 / 2) * \
        #      np.exp(-(x[i, :] - mu1).dot(np.linalg.inv(sigma).dot((x[i, :] - mu1).T)) / 2) * phi
        #
        # res = np.zeros(x.shape[0])
        # for i in range(x.shape[0]):
        #     if p0[i] > p1[i]:
        #         res[i] = 0
        #     else:
        #         res[i] = 1
        # return res
        res = 1 / (1 + np.exp(-x.dot(self.theta)))
        for i in range(x.shape[0]):
            if res[i] >= 0.5:
                res[i] = 1
            else:
                res[i] = 0
        return res


if __name__ == "__main__":
    main("data/ds2_train.csv", "data/ds2_valid.csv", "data.ds2_output_gda")
