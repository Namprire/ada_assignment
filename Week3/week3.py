import numpy as np
import matplotlib.pyplot as plt

def generate_sample(xmin=-3, xmax=3, sample_size=50):
    x = np.linspace(xmin, xmax, sample_size)
    pix = np.pi * x
    y_true = np.sinc(x) + 0.1 * x
    noise = 0.05 * np.random.randn(sample_size)
    return x, y_true + noise

def calc_design_matrix(x, c, h):
    return np.exp(-(x[:, None] - c[None, :])**2 / (2 * h**2))

def l1_iterative_reweighted_least_squares(K, y, lamb=0.1, tol=1e-6, max_iter=100):
    n = K.shape[1]
    theta = np.linalg.solve(K.T @ K + lamb * np.eye(n), K.T @ y)
    for _ in range(max_iter):
        w = np.diag(1 / (np.abs(theta) + 1e-8))  # prevent div by zero
        theta_new = np.linalg.solve(K.T @ K + lamb * w, K.T @ y)
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta

x_train, y_train = generate_sample()
h = 0.2
K = calc_design_matrix(x_train, x_train, h)
theta = l1_iterative_reweighted_least_squares(K, y_train)

# Prediction
x_test = np.linspace(-3, 3, 1000)
K_test = calc_design_matrix(x_test, x_train, h)
y_pred = K_test @ theta

# Plot
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_test, y_pred, color='green', linewidth=2)
plt.title("ℓ₁-constrained LS Regression")
plt.show()
