import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 2D-->1D
"""
mat = loadmat('ex7data1.mat')
print(mat)
X = mat['X']"""
X = np.array([[1.0,2],[2,4]])
print('X.shape:',X.shape)  # (50, 2)
print(X)
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')

# X 均值归一化
def featureNormal(X):
    means = X.mean(axis=0)
    print('means：',means)
    stds = X.std(axis=0, ddof=1)
    print('stds:',stds)
    X_norm = (X - means) / stds
    print('x_norm:',X_norm)
    return X_norm, means, stds


# PCA
def pca(X):
    sigma = (X.T @ X) / len(X)
    print("sigma:",sigma)
    U, S, V = np.linalg.svd(sigma)
    print('U',U)
    print('S',S)
    print('V',V)
    return U, S, V


X_norm, means, stds = featureNormal(X)
print("std:",X_norm)
print("std:",means)
print("std:",stds)
U, S, V = pca(X_norm)

print('U:',U[:, 0])

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.plot([means[0], means[0] + 1.5 * S[0] * U[0, 0]],
         [means[1], means[1] + 1.5 * S[0] * U[0, 1]],
         c='r', linewidth=3, label='First Principal Component')

plt.plot([means[0], means[0] + 1.5 * S[1] * U[1, 0]],
         [means[1], means[1] + 1.5 * S[1] * U[1, 1]],
         c='g', linewidth=3, label='Second Principal Component')
plt.grid()
plt.axis("equal")
plt.legend()


# Dimensionality Reduction with PCA


def projectData(X, U, K):
    Z = X @ U[:, :K]
    print(K)
    print(Z)
    print('hahah')
    return Z


Z = projectData(X_norm, U, 1)
Z[0]


# print(Z[0]) #[ 1.48127391]


# Reconstructing an approximation of the data 重建数据
def recoverData(Z, U, K):
    X_rec = Z @ U[:, :K].T
    return X_rec


X_rec = recoverData(Z, U, 1)
X_rec[0]
# print(X_rec[0])     #[-1.04741883 -1.04741883]


# Visualizing the projections
plt.figure(figsize=(7, 5))
plt.axis("equal")
plot = plt.scatter(X_norm[:, 0], X_norm[:, 1], s=30, facecolors='none',
                   edgecolors='b', label='Original Data Points')

plot = plt.scatter(X_rec[:, 0], X_rec[:, 1], s=30, facecolors='none',
                   edgecolors='r', label='PCA Reduced Data Points')

plt.title("Example Dataset: Reduced Dimension Points Shown", fontsize=14)
plt.xlabel('x1 [Feature Normalized]', fontsize=14)
plt.ylabel('x2 [Feature Normalized]', fontsize=14)
plt.grid(True)

for x in range(X_norm.shape[0]):
    plt.plot([X_norm[x, 0], X_rec[x, 0]], [X_norm[x, 1], X_rec[x, 1]], 'k--')
    # 输入第一项全是X坐标 第二项全是y坐标
plt.legend()
plt.show()
