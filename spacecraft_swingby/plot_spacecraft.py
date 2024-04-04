import numpy as np
import torch
import matplotlib.pyplot as plt
from pinn_sc import Net


def plot_trajectory_force(filename, Tvalue = None):
    # load data
    # filename = r'data/model_20240327-232831.pt'
    checkpoint = torch.load(filename)
    net = Net([1] + [64] * 3 + [2])
    net.load_state_dict(checkpoint['model_state_dict'])

    # calculate the solution
    n = 1001
    dt = 1 / (n - 1)
    tmin, tmax = 0.0, 1.0
    t = torch.linspace(tmin, tmax, n, requires_grad=True).reshape((n, 1))
    m0 = 1.
    bh_xygm = [
        [-0.5, -1.0, 0.5],
        [-0.2, 0.4, 1.0],
        [0.8, 0.3, 0.5],
    ]

    if Tvalue:
        T = Tvalue
    else:
        with open('data/T.txt', 'r') as f:
            T = float(f.read())

    u = net(t).detach().numpy()
    x, y = u[:, 0], u[:, 1]
    xt = np.gradient(x) / dt / T
    xtt = np.gradient(xt) / dt / T
    yt = np.gradient(y) / dt / T
    ytt = np.gradient(yt) / dt / T


    fgx, fgy = [], []
    for xtmp, ytmp, gmtmp in bh_xygm:
        fgx.append(-gmtmp * m0 * (x - xtmp) / ((x - xtmp) ** 2 + (y - ytmp) ** 2) ** 1.5)
        fgy.append(-gmtmp * m0 * (y - ytmp) / ((x - xtmp) ** 2 + (y - ytmp) ** 2) ** 1.5)
    fgx.append(-xtt)
    fgy.append(-ytt)
    fgx = np.array(fgx)
    fgy = np.array(fgy)


    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(x,y, label="Trajectory")
    for i, (xtmp, ytmp, gmtmp) in enumerate(bh_xygm):
        ax.scatter(xtmp, ytmp, s=gmtmp * 500, c='r', marker='o')
    ax.quiver(x[::50][1:-1],
              y[::50][1:-1],
              np.sum(fgx[:-1], axis=0)[::50][1:-1],
              np.sum(fgy[:-1], axis=0)[::50][1:-1],
              color='gray', scale=20., width=0.01, label='Gravity/20')
    ax.quiver(x[::50][1:-1],
              y[::50][1:-1],
              np.sum(fgx, axis=0)[::50][1:-1],
              np.sum(fgy, axis=0)[::50][1:-1],
              color='orange', scale=1., width=0.01, label='Thrust')
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.grid()
    plt.legend()
    plt.savefig(f'data/trajectory2.png', dpi=300)
    plt.show()


    t = t.detach().numpy()

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    cs = ['r', 'k', 'g']
    for i, (xtmp, ytmp, gmtmp) in enumerate(bh_xygm):
        ax.plot(t,
                np.sqrt(fgx[i] ** 2 + fgy[i] ** 2),
                lw=1, c=cs[i], label=f'Gravity by Object {i+1}')

    ax.plot(t, np.sqrt(np.sum(fgx[:-1], axis=0) ** 2 + np.sum(fgy[:-1], axis=0) ** 2), lw=4, c='gray', label='Total gravity')
    ax.plot(t[2:-2], np.sqrt(fgx[-1] ** 2 + fgy[-1] ** 2)[2:-2], lw=1.5, c='b', ls='--', label='Required force for the trajectory')
    ax.plot(t[2:-2], np.sqrt(np.sum(fgx, axis=0) ** 2 + np.sum(fgy, axis=0) ** 2)[2:-2], lw=2, c='orange', label='Thrust')
    plt.legend(ncols=2, fontsize=9.5)
    plt.savefig(f'data/forces2.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # plot_trajectory_force(r'data/model_20240328-143949.pt', Tvalue=2.0269)
    plot_trajectory_force(r'data/model_20240328-182213.pt', Tvalue=2.0988)

