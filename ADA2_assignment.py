
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import product


np.random.seed(0)

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(xmin, xmax, sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(size=sample_size)
    return x, target + noise

def calc_design_matrix(x_basis, x_sample, h):
    return np.exp(-((x_sample[:, None] - x_basis[None, :]) ** 2) / (2 * h ** 2))

def cross_validate_kernel_ridge(x, y, lambda_vals, h_vals, k=5):
    results = {}
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    for lam, h in product(lambda_vals, h_vals):
        mse_list = []
        for train_idx, val_idx in kf.split(x):
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            K_train = calc_design_matrix(x_train, x_train, h)
            K_val = calc_design_matrix(x_train, x_val, h)

            theta = np.linalg.solve(
                K_train + lam * np.identity(len(K_train)),
                y_train[:, None]
            )

            y_pred = K_val @ theta
            mse_list.append(mean_squared_error(y_val, y_pred.ravel()))

        results[(lam, h)] = np.mean(mse_list)
    return results

def visualize_kernel_regression(x, y, best_lambda, best_h, xmin, xmax):
    K_train = calc_design_matrix(x, x, best_h)
    theta = np.linalg.solve(K_train + best_lambda * np.identity(len(K_train)), y[:, None])

    X_plot = np.linspace(xmin, xmax, 500)
    K_plot = calc_design_matrix(x, X_plot, best_h)
    y_plot = K_plot @ theta

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='green', label='Training data')
    plt.plot(X_plot, y_plot, color='blue', label='Kernel regression')
    plt.title("L2-Regularized Kernel Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    sample_size = 50
    xmin, xmax = -3, 3
    x, y = generate_sample(xmin, xmax, sample_size)

    lambda_vals = np.logspace(-3, 1, 5)
    h_vals = np.linspace(0.05, 0.5, 5)

    results = cross_validate_kernel_ridge(x, y, lambda_vals, h_vals)
    best_params = min(results, key=results.get)
    best_lambda, best_h = best_params

    print(f"Best λ: {best_lambda}, Best h: {best_h}, CV MSE: {results[best_params]:.5f}")
    visualize_kernel_regression(x, y, best_lambda, best_h, xmin, xmax)
