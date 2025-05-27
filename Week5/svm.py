import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
np.random.seed(1)

def generate_data(sample_size):
    n = sample_size

    # Replicate MATLAB: x = [randn(1,n/2)-5, randn(1,n/2)+5; randn(1,n)]
    x1 = np.concatenate([np.random.randn(n//2) - 5, np.random.randn(n//2) + 5])
    x2 = np.random.randn(n)
    x = np.stack([x1, x2], axis=1)

    # Add bias term
    x = np.hstack([x, np.ones((n, 1))])

    # Labels: first half +1, second half -1
    y = np.concatenate([np.ones(n//2), -np.ones(n//2)])

    # Perturb margin violators
    x[:3, 1] -= 5
    y[:3] = -1
    x[n//2:n//2+3, 1] += 5
    y[n//2:n//2+3] = 1

    return x, y



def svm(x, y, l, lr):
    w = np.zeros(3)
    prev_w = w.copy()
    for _ in range(10**4):
        idx = np.random.randint(0, len(x))
        x_i = x[idx]
        y_i = y[idx]
        margin = y_i * np.dot(w, x_i)

        if margin < 1:
            grad = l * w - y_i * x_i
        else:
            grad = l * w

        w -= lr * grad

        if np.linalg.norm(w - prev_w) < 1e-3:
            break
        prev_w = w.copy()
    return w

def visualize(x, y, w):
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='blue', label='Class +1')
    plt.scatter(x[y == -1, 0], x[y == -1, 1], c='red', label='Class -1')

    # Plot decision boundary
    x_vals = np.array([-10, 10])
    if abs(w[1]) > 1e-6:
        y_vals = -(w[2] + w[0] * x_vals) / w[1]
        plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')
    else:
        print("Warning: w[1] ≈ 0, can't plot line.")

    plt.legend()
    plt.title("Linear SVM Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

# Run
x, y = generate_data(200)
w = svm(x, y, l=0.01, lr=0.01)  # Tuned values for good separation
visualize(x, y, w)
