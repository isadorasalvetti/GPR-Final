import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotSurface():
    X = np.arange(1, 2, 0.1)
    Y = np.arange(0, 2, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = (np.sin(np.pi*X)*np.cos(np.pi*X)) / (1 + X*X*Y*Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()


def plotPointsPlane(p, fit, rp):
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    xs, ys, zs = [], [], []
    for point in p:
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])

    ax.scatter(xs, ys, zs, color='b')
    ax.scatter(rp[0], rp[1], rp[2], color='r')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                       np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)

    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]
    ax.plot_wireframe(X, Y, Z, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def plotSolution(p, rp, s):
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    xs, ys, zs = [], [], []
    for point in p:
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])

    ax.scatter(xs, ys, zs, color='b')
    ax.scatter(rp[0], rp[1], rp[2], color='r')
    ax.scatter(s[0], s[1], s[2], color='g')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

