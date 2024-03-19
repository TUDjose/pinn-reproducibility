import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)
        ])
        self.activation = torch.tanh

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.linears[:-2]:
            x = self.activation(layer(x))
        return self.linears[-1](x)


tmin, tmax = 0.0, 1.0
x0, y0 = -1., -1.
x1, y1 = 1., 1.
m0 = 1.
bh_xygm = [
    [-0.5, -1.0, 0.5],
    [-0.2, 0.4, 1.0],
    [0.8, 0.3, 0.5],
]

n_output = 2
n_adam = 2000
n_domain = 1000
xT, xTT = None, None
yT, yTT = None, None


def constraint_loss(x, y, x_T, x_TT, y_T, y_TT, t):
    return 10 * ((x[0] - x0) ** 2 + (y[0] - y0) ** 2 + (x[-1] - x1) ** 2 + (y[-1] - y1) ** 2)

def physics_loss(x, y, x_T, x_TT, y_T, y_TT, t):
    ode_x = x_TT
    ode_y = y_TT
    for xtmp, ytmp, gmtmp in bh_xygm:
        ode_x += gmtmp * m0 * (x.reshape((-1, 1)) - xtmp) / ((x.reshape((-1, 1)) - xtmp) ** 2 + (y.reshape((-1, 1)) - ytmp) ** 2) ** 1.5
        ode_y += gmtmp * m0 * (y.reshape((-1, 1)) - ytmp) / ((x.reshape((-1, 1)) - xtmp) ** 2 + (y.reshape((-1, 1)) - ytmp) ** 2) ** 1.5
    return torch.mean(torch.norm(ode_x, p=2)**2) + torch.mean(torch.norm(ode_y, p=2)**2)

def pinn_loss(x, y, t, loss_weights, loss_fns):
    global xT, xTT, yT, yTT
    x_T = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0].reshape((-1, 1))
    y_T = grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0].reshape((-1, 1))
    x_TT = grad(x_T, t, grad_outputs=torch.ones_like(x_T), create_graph=True)[0].reshape((-1, 1))
    y_TT = grad(y_T, t, grad_outputs=torch.ones_like(y_T), create_graph=True)[0].reshape((-1, 1))

    return sum([w * L(x, y, x_T, x_TT, y_T, y_TT, t) for w, L in zip(loss_weights, loss_fns)])


net = Net([1] + [64] * 3 + [n_output])
time_domain = torch.linspace(tmin, tmax, n_domain).reshape((n_domain, 1))
time_domain.requires_grad = True

def train(model, optimizer, steps):
    model.train()

    for n in range(steps):
        u = model(time_domain)
        x, y = u[:, 0], u[:, 1]
        loss = pinn_loss(x, y, time_domain, [1., 1.], [constraint_loss, physics_loss])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if n % 100 == 0:
            print(loss)

def plot(model):
    model.eval()
    u = model(time_domain)
    x, y = u[:, 0], u[:, 1]
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.show()

print("Training..")
train(net, torch.optim.Adam(net.parameters(), lr=1e-3), n_adam)

print("Plotting..")
plot(net)
