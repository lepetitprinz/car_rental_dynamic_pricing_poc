import os
import numpy as np


class OfflineLearning(object):
    def __init__(self):
        self.load_path = os.path.join('..', 'result')

    def _cost_func(self, x: np.array, y: np.array, theta: float):

        m = len(y)
        rec = self._rec_disc(theta=theta, curr=x[:, 0], exp=x[:, 1])
        lower = self._lower_bound(disc=x[:, 2], y=y) - rec
        upper = rec - self._upper_bound(disc=x[:, 2], y=y)
        cost = (1 / m) * np.sum(self.relu(lower), self.relu(upper))

        return cost

    def stochastic_gradient_descent(self, x: np.array, y: np.array, theta: float,
                                    learning_rate=0.01, iterations=100):
        """
        :param x: X: Matrix of X with added bias units
        :param y: y: Vector of Y
        :param theta: Vector o f thetas np.random.randn(j, 1)
        :param learning_rate: Learning rate
        :param iterations: # of iterations
        :return:
        """
        m = len(y)
        cost_history = np.zeros(iterations)

        for it in range(iterations):
            cost = 0.0
            for i in range(m):
                rand_ind = np.random.randint(0, m)
                X_i = x[rand_ind, :].reshape(1, x.shape[1])
                y_i = y[rand_ind].reshape(1, 1)
                prediction = np.dot(X_i, theta)

                theta = theta - (1 / m) * learning_rate * (X_i.T.dot((prediction - y_i)))
                cost += self._cost_func(theta, X_i, y_i)
            cost_history[it] = cost

    @staticmethod
    def _lower_bound(disc: np.array, y: np.array):
        c_1 = 0.5
        return np.multiply(y, disc) + np.multiply(1 - y, c_1, disc)

    @staticmethod
    def _upper_bound(disc: np.array, y: np.array):
        c_2 = 2
        return np.multiply(1 - y, disc) + np.multiply(y, c_2, disc)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def calc_cost(theta, x, y):
        m = len(y)

        predictions = x.dot(theta)
        cost = (1 / 2 * m) * np.sum(np.square(predictions - y))

        return cost

    @staticmethod
    def _rec_disc(theta: float, curr: float, exp: float):
        """
        Recommendation Objective function
        curr: Current Utilization Rate (0 < curr < 1)
        exp: Expected Utilization rate (0 < exp < 1)
        """
        # Hyper-parameters (need to tune)
        if curr < exp:  # Ratio of increasing magnitude
            theta = 0.7 * theta

        # Suggestion curve bend
        # 1 < phi_low < phi_high < 2
        phi_high = 1.2

        y = 1 - theta * (curr ** (phi_high ** (-1 * curr)) - exp)

        return y

    def _gradient_descent_BAK(self, x, y, theta, learning_rate=0.01, iterations=100):
        """
        :param x: Matrix of X with added bias units
        :param y: Vector of Y
        :param theta: Vector of thetas np.random.randn(j, 1)
        :param learning_rate: Learning rate
        :param iterations: # of iterations
        :return:
        """
        m = len(y)
        cost_history = np.zeros(iterations)
        theta_history = np.zeros((iterations, 2))
        for it in range(iterations):
            prediction = np.dot(x, theta)

            theta = theta - (1 / m) * learning_rate * (x.T.dot((prediction - y)))
            theta_history[it, :] = theta.T
            cost_history[it] = self.calc_cost(theta, x, y)

        return theta, cost_history, theta_history

    def _stochastic_gradient_descent_BAK(self, x, y, theta, learning_rate=0.01, iterations=100):
        """
        :param x: X: Matrix of X with added bias units
        :param y: y: Vector of Y
        :param theta: Vector o f thetas np.random.randn(j, 1)
        :param learning_rate: Learning rate
        :param iterations: # of iterations
        :return:
        """
        m = len(y)
        cost_history = np.zeros(iterations)

        for it in range(iterations):
            cost = 0.0
            for i in range(m):
                rand_ind = np.random.randint(0, m)
                X_i = x[rand_ind, :].reshape(1, x.shape[1])
                y_i = y[rand_ind].reshape(1, 1)
                prediction = np.dot(X_i, theta)

                theta = theta - (1 / m) * learning_rate * (X_i.T.dot((prediction - y_i)))
                cost += self.calc_cost(theta, X_i, y_i)
            cost_history[it] = cost
