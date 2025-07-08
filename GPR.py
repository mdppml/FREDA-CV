import numpy as np
from scipy.optimize import minimize


def linear_kernel(X1, X2=None, variance=1.0):
    if X2 is None:
        X2 = X1
    return variance * (X1 @ X2.T)


class GPR:
    def __init__(self, X: np.array = None, Y: np.array = None, kernel_var=1.0, noise=1.0):
        self.X = X
        self.Y = Y
        self.kernel_var = kernel_var
        self.noise = noise

    def negative_log_marginal_likelihood(self, log_params):
        log_kernel_var, log_noise = log_params
        kernel_var = np.exp(log_kernel_var)
        noise = np.exp(log_noise)

        K = linear_kernel(self.X, self.X, kernel_var) + noise * np.eye(len(self.X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.Y))

        log_likelihood = -0.5 * self.Y.T @ alpha
        log_likelihood -= np.sum(np.log(np.diagonal(L)))
        log_likelihood -= 0.5 * len(self.X) * np.log(2 * np.pi)

        return -log_likelihood.flatten()[0]

    def optimize(self):
        res = minimize(self.negative_log_marginal_likelihood,
                       x0=np.log([self.kernel_var, self.noise]),
                       bounds=[(-5, 5), (-5, 5)],
                       method='L-BFGS-B')

        self.kernel_var = np.exp(res.x[0])
        self.noise = np.exp(res.x[1])
        return self.kernel_var, self.noise

    def predict(self, x_new):
        K = linear_kernel(self.X, self.X, self.kernel_var)
        K_star = linear_kernel(x_new, self.X, self.kernel_var)
        K_star_star = linear_kernel(x_new, x_new, self.kernel_var)

        K += self.noise * np.eye(len(self.X))
        inv_K = np.linalg.inv(K)

        mean = K_star @ inv_K @ self.Y
        var_diag = np.diag(K_star_star) - np.sum(K_star @ inv_K * K_star, axis=1)
        std = np.sqrt(np.maximum(var_diag, 1e-10))

        return mean.flatten(), std

    def predict_from_matrices(self, K, K_star, K_star_star):
        inv_K = np.linalg.inv(K + self.noise * np.eye(len(self.X)))

        mean = K_star @ inv_K @ self.Y
        var_diag = np.diag(K_star_star) - np.sum(K_star @ inv_K * K_star, axis=1)
        std = np.sqrt(np.maximum(var_diag, 1e-10))

        return mean.flatten(), std
