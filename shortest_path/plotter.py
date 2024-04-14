import numpy as np
import torch
import matplotlib.pyplot as plt
from light import Net as LNet
from brachistochrone import Net as BNet


def plot_trajectory_light(filename):
    checkpoint = torch.load(filename)
    net = LNet([1] + [64] * 3 + [2])
    net.load_state_dict(checkpoint['model_state_dict'])

    n = 1001
    tmin, tmax = 0.0, 1.0
    t = torch.linspace(tmin, tmax, n, requires_grad=True).reshape((n, 1))
    c, n1, n2 = 1., 1., 2.

    u = net(t).detach().numpy()
    x, y = u[:, 0], u[:, 1]

    yy = np.linspace(0, 1, 1001)
    xx = np.arctan(2. * np.tan(np.pi * yy)) / np.pi
    xx[len(xx) // 2 + 1:] += 1

    X, Y = np.meshgrid(xx, yy)
    R = n1 + (n2 - n1) * 0.5 * (1. - np.cos(2. * np.pi * Y))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    z = ax.contourf(X, Y, R, levels=30, alpha=0.5, zorder=-1)
    cbar = plt.colorbar(z)
    ax.plot(x, y, label="Trajectory")
    ax.plot(xx, yy, label="Analytical")

    cbar.ax.set_ylabel("Refraction index")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("data/trajectory.png", dpi=300)
    plt.show()


def plot_trajectory_brachistochrone(filename):
    checkpoint = torch.load(filename)
    net = BNet([1] + [64] * 3 + [2])
    net.load_state_dict(checkpoint['model_state_dict'])

    n = 1001
    dt = 1 / (n - 1)
    tmin, tmax = 0.0, 1.0
    t = torch.linspace(tmin, tmax, n, requires_grad=True).reshape((n, 1))

    u = net(t).detach().numpy()
    x, y = u[:, 0], u[:, 1]

    r = 0.5729
    theta = np.linspace(0, np.arccos(1 - 1 / r), 1001)
    xx = r * (theta - np.sin(theta))
    yy = 1 - r * (1 - np.cos(theta))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(x, y, label="Trajectory")
    ax.plot(xx, yy, label="Analytic")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.legend()
    plt.grid()
    plt.savefig("data/trajectory_b.png", dpi=300)
    plt.show()



if __name__ == '__main__':
    # plot_trajectory_light('data/checkpoint.pt')
    plot_trajectory_brachistochrone('data/checkpoint.pt')
