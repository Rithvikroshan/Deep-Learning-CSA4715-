import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    mean_x, mean_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x - n * mean_y * mean_x)
    SS_xx = np.sum(x * x - n * mean_x * mean_x)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1 * mean_x

    return b_0, b_1

def plot_regression_line(x, y, b):
    # plotting the actual points
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # function to show plot
    plt.show()

def main():
    # generate a linearly separable dataset
    np.random.seed(42)
    x_positive = np.random.normal(5, 1, 50)
    y_positive = np.random.normal(5, 1, 50)

    x_negative = np.random.normal(10, 1, 50)
    y_negative = np.random.normal(10, 1, 50)

    # combine positive and negative classes
    x = np.concatenate([x_positive, x_negative])
    y = np.concatenate([y_positive, y_negative])

    # estimating coefficients
    b = estimate_coef(x, y)

    # plotting regression line
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()
