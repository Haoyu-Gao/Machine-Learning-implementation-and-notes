import matplotlib.pyplot as plt
import numpy as np
from regression import util
from regression.weighted_regression import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """ Tune the bandwidth parameter tau for LWR

    :param tau_values: List of tau values to try
    :param train_path: Path to CSV file containing training set.
    :param valid_path: Path to CSV file containing validation set.
    :param test: Path to CSV file containing test set.
    :param pred_path: Path to save predictions
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)

    # Search tau_values for the best tau (lowest MSE on the validation set)
    mse_list = []
    for tau in tau_values:
        model.tau = tau
        y_pred = model.predict(x_eval)

        mse = np.mean((y_pred - y_eval) ** 2)
        mse_list.append(mse)
        print(f'valid set: tau={tau}, MSE={mse}')

        plt.figure()
        plt.title('tau = {}'.format(tau))
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_eval, y_pred, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('data/ds5_c_tau_{}.png'.format(tau))

    # Fit a LWR model with the best tau value
    tau_opt = tau_values[np.argmin(mse_list)]
    print(f'valid set: lowest MSE={min(mse_list)}, tau={tau_opt}')

    y_pred = model.predict(x_test)
    np.savetxt(pred_path, y_pred)

    mse = np.mean((y_pred - y_test) **2)
    print(f'test set: tau{tau_opt}, MSE={mse}')
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data


if __name__ == '__main__':
    main([0.01, 0.03, 0.05, 0.07, 0.09], "data/ds5_train.csv", "data/ds5_valid.csv", "data/ds5_test.csv", "data/ds5_pred_C.csv")

