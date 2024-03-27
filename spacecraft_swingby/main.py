import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.activation = torch.tanh

        # self.T = torch.autograd.Variable(torch.tensor([1], dtype=torch.float32).to(device), requires_grad=True)

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return x


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
n_adam = 5000
n_lbfgs = 4000
n_domain = 1000
lr_adam, lr_lbfgs = 4e-3, 3.8e-3
xT, xTT = None, None
yT, yTT = None, None

T = torch.autograd.Variable(torch.tensor([1], dtype=torch.float32).to(device), requires_grad=True)

weights = [10., 1.]

net = Net([1] + [64] * 3 + [n_output])
time_domain = torch.linspace(tmin, tmax, n_domain, requires_grad=True).reshape((n_domain, 1)).to(device)

def constraint_loss(x, y, x_T, x_TT, y_T, y_TT, t):
    return ((x[0] - x0) ** 2 + (y[0] - y0) ** 2 + (x[-1] - x1) ** 2 + (y[-1] - y1) ** 2)

def physics_loss(x, y, x_T, x_TT, y_T, y_TT, t):
    ode_x = x_TT
    ode_y = y_TT
    for xtmp, ytmp, gmtmp in bh_xygm:
        ode_x += gmtmp * m0 * (x.reshape((-1, 1)) - xtmp) / ((x.reshape((-1, 1)) - xtmp) ** 2 + (y.reshape((-1, 1)) - ytmp) ** 2) ** 1.5
        ode_y += gmtmp * m0 * (y.reshape((-1, 1)) - ytmp) / ((x.reshape((-1, 1)) - xtmp) ** 2 + (y.reshape((-1, 1)) - ytmp) ** 2) ** 1.5

    return torch.mean(torch.norm(ode_x, p=2)**2) + torch.mean(torch.norm(ode_y, p=2)**2)

def pinn_loss(x, y, t, loss_weights, loss_fns):
    global xT, xTT, yT, yTT
    x_T = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0].reshape((-1, 1)).to(device) / T
    y_T = grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0].reshape((-1, 1)).to(device) / T
    x_TT = grad(x_T, t, grad_outputs=torch.ones_like(x_T), create_graph=True)[0].reshape((-1, 1)).to(device) / T
    y_TT = grad(y_T, t, grad_outputs=torch.ones_like(y_T), create_graph=True)[0].reshape((-1, 1)).to(device) / T

    return sum([w * L(x, y, x_T, x_TT, y_T, y_TT, t) for w, L in zip(loss_weights, loss_fns)])


def train(model, optimizer, steps):
    model.train()

    for n in range(steps):
        u = model(time_domain)
        x, y = u[:, 0], u[:, 1]
        loss = pinn_loss(x, y, time_domain, weights, [constraint_loss, physics_loss])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if n % 100 == 0:
            print(f"Step {n}, Loss: {loss.item()}, T: {model.T.item()}")


    def closure():
        optimizer.zero_grad()
        u = model(time_domain)
        x, y = u[:, 0], u[:, 1]
        loss = pinn_loss(x, y, time_domain, loss_weights=weights, loss_fns=[constraint_loss, physics_loss])
        loss.backward()
        return loss

    print("L-BFGS..")
    optimizer = torch.optim.LBFGS(list(model.parameters() + [model.T]), lr=lr_lbfgs)
    try:
        for n in range(n_lbfgs):
            optimizer.step(closure)
            loss = closure()
            if n % 100 == 0:
                print(f"Step {n}, Loss: {loss.item()}, T: {model.T.item()}")
    except KeyboardInterrupt:
        pass

    torch.save({
            'epoch': n,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'data/model_{time.strftime("%Y%m%d-%H%M%S")}.pt')


def plot(model):
    model.eval()
    with torch.no_grad():
        u = [model(t.reshape(1)).to(device) for t in time_domain]
    x = torch.stack([u[i][0] for i in range(n_domain)]).cpu()
    y = torch.stack([u[i][1] for i in range(n_domain)]).cpu()
    plt.figure(figsize=(6, 6))
    plt.plot(x.detach().numpy(), y.detach().numpy())
    for i, (xtmp, ytmp, gmtmp) in enumerate(bh_xygm):
        plt.scatter(xtmp, ytmp, s=gmtmp * 500, c='r', marker='o')
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.grid()
    plt.savefig(f'plot_{time.strftime("%Y%m%d-%H%M%S")}.png', dpi=300)
    plt.show()


def retrain(model, optimizer, steps):
    model.train()

    def closure():
        optimizer.zero_grad()
        u = model(time_domain)
        x, y = u[:, 0], u[:, 1]
        loss = pinn_loss(x, y, time_domain, loss_weights=weights, loss_fns=[constraint_loss, physics_loss])
        loss.backward()
        return loss

    print("L-BFGS part 2 ...")
    try:
        for n in range(steps):
            optimizer.step(closure)
            loss = closure()
            if n % 100 == 0:
                print(loss, n, T)
    except KeyboardInterrupt:
        pass

    torch.save({
        'epoch': n,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'data/model_{time.strftime("%Y%m%d-%H%M%S")}.pt')


if __name__ == '__main__':
    print("Training..")
    optimizer = torch.optim.Adam([
        {'params': net.parameters()}], lr=lr_adam)
    net.to(device)
    train(net, optimizer, n_adam)

    print("Plotting..")
    plot(net)

    # print("Retraining...")
    # checkpoint = torch.load('model_20210827-173104.pt')
    # net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = torch.optim.LBFGS(list(net.parameters()) + [T], lr=0.1)
    # retrain(net, optimizer, 6000)
    # plot(net)
    #
    # print("Retraining...")
    # checkpoint = torch.load('model_20210827-173104.pt')
    # net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = torch.optim.LBFGS(list(net.parameters()) + [T], lr=0.5)
    # retrain(net, optimizer, 6000)
    # plot(net)
