import numpy as np
from regression import util

from regression.logistic_regiression import LogisticRegression

# Character to replace with sub-problem letter in plot_path/ pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """ Logistic regression for incomplete, positive-only labels

    Run under the following conditions:
        1. on y-labels
        2. on l-labels
        3 on l-labels with correction factor alpha

    :param train_path: Path to CSV file containing training set
    :param valid_path: Path to CSV file containing validation set
    :param test_path: Path to CSV file containing test set
    :param pred_path: Path to save predictions
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    #####################################
    # Problem C code
    x_train, t_train = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    log_reg_model = LogisticRegression()
    log_reg_model.fit(x_train, t_train)

    util.plot(x_test, t_test, log_reg_model.theta, "data/ds3_test_C")
    t_predict_c = log_reg_model.predict(x_test)
    np.savetxt(pred_path_c, t_predict_c > 0.5, fmt="%d")

    ####################################
    # Problem D code
    x_train_d, y_train_d = util.load_dataset(train_path, add_intercept=True)
    x_test_d, y_test_d = util.load_dataset(test_path, add_intercept=True)

    log_reg_model_d = LogisticRegression()
    log_reg_model_d.fit(x_train_d, y_train_d)

    util.plot(x_test_d, y_test_d, log_reg_model_d.theta, "data/ds3_test_D")
    y_predict_d = log_reg_model_d.predict(x_test_d)
    np.savetxt(pred_path_c, t_predict_c > 0.5, fmt="%d")
    # It is clear that the partial positive only problem should be done in other methods!

    #####################################
    # Problem E code
    x_valid, y_valid = util.load_dataset(valid_path, label_col="y", add_intercept=True)

    alpha = np.mean(log_reg_model_d.predict(x_valid[y_valid == 1]))

    correction = 1 + np.log(2 / alpha - 1) / log_reg_model_d.theta[0]
    util.plot(x_test_d, t_test, log_reg_model_d.theta, "data/ds3_e", correction)

    t_pred_e = y_predict_d / alpha
    np.savetxt(pred_path_e, t_pred_e > 0.5, fmt="%d")


if __name__ == '__main__':
    main("data/ds3_train.csv", "data/ds3_valid.csv", "data/ds3_test.csv", "data/ds3_pred_X.csv")