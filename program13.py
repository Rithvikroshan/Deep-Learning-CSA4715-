import numpy as np

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

def compute_loss(X, y, w, b):
    N = len(y)
    predictions = np.dot(X, w) + b
    loss = np.sum((predictions - y) ** 2) / N
    return loss

def gradient_descent(X, y, learning_rate, iterations, stopping_threshold):
    w = np.random.randn(X.shape[1], 1)
    b = np.random.randn()

    for i in range(iterations):
        predictions = np.dot(X, w) + b
        dw = (1 / len(y)) * np.dot(X.T, (predictions - y))
        db = (1 / len(y)) * np.sum(predictions - y)
        w -= learning_rate * dw
        b -= learning_rate * db
        loss = compute_loss(X, y, w, b)
        if loss < stopping_threshold:
            break
    return w, b

learning_rate = 0.01
iterations = 1000
stopping_threshold = 0.01
optimal_w, optimal_b = gradient_descent(X, y, learning_rate, iterations, stopping_threshold)

print("Optimal parameters:")
print("Weight:", optimal_w)
print("Bias:", optimal_b)
