import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


input_file='dataset.tsv'
output_file='out_dataset.tsv'


def load_data(filename="dataset.tsv"):
    df = pd.read_csv(filename, sep="\t")
    X = df[["X1", "X2"]].values
    Y0 = df["Y0"].values
    return X, Y0


def sgd_with_constraints(X, Y0, beta_0, lr=0.01, max_iters=1000, tol=1e-6):
    n = len(Y0)
    Y = Y0.copy()
    
    A = np.linalg.inv(X.T @ X) @ X.T
    
    for it in range(max_iters):
        grad = 2 * (Y - Y0)
        Y_new = Y - lr * grad
        AY = A @ Y_new
        correction = A.T @ np.linalg.inv(A @ A.T) @ (AY - beta_0)
        Y_new = Y_new - correction
        if np.linalg.norm(Y_new - Y) < tol:
            break
        Y = Y_new
    
    return Y


def draw_graph(X, Y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color="blue", marker="o", label="Точки")
    plt.plot(X, Y, color="lightblue", linestyle="--", alpha=0.6)
    plt.show()


if __name__ == "__main__":
    X, Y0 = load_data("dataset.tsv")
    beta0 = np.array([1.0, 0.9])
    Y_opt = sgd_with_constraints(X, Y0, beta0, lr=0.01, max_iters=5000)
    
    draw_graph([i[0] for i in X], Y_opt)    
