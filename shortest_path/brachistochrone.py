import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
import random
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.activation = torch.tanh

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.linears[:-2]:
            x = self.activation(layer(x))
        return torch.sigmoid_(self.linears[-1](x))


class SPBPINN():
    def __init__(self):
        self.tmin, self.tmax = 0., 1.
        self.x0, self.x1 = 0., 1.
        self.y0, self.y1 = 1., 0.
        self.g = 9.8

        self.n_output = 2
        self.n_adam = 2000
        self.n_lbfgs = 1500
        self.n_domain = 1000
        self.lr_adam, self.lr_lbfgs = 3e-3, 1e-2
        self.xT, self.yT = None, None

        self.weights = [1., 1., 0.]

        self.net = Net([1] + [64] * 3 + [self.n_output])
        self.net.to(device)
        self.time_domain = torch.linspace(self.tmin, self.tmax, self.n_domain, requires_grad=True).reshape((self.n_domain, 1)).to(device)
        self.T = torch.autograd.Variable(torch.tensor([1.], dtype=torch.float32).to(device), requires_grad=True)

        self.ploss = []
        self.closs = []
        self.gloss = []

    def pde_resampler(self):
        points = [self.tmin]
        points += [random.uniform(self.tmin, self.tmax) for _ in range(self.n_domain - 2)]
        points += [self.tmax]
        points.sort()

        points_tensor = torch.tensor(points).reshape((self.n_domain, 1)).to(device)
        points_tensor.requires_grad = True

        return points_tensor

    def constraint_loss(self, x, y, x_T, y_T, t):
        L = (x[0]) ** 2 + (y[0] - 1.) ** 2 + (x[-1] - 1.) ** 2 + (y[-1]) ** 2
        self.closs.append(L.item())
        return L

    def physics_loss(self, x, y, x_T, y_T, t):
        ode = self.g * self.y0 - (self.g * y + 0.5 * ((x_T / self.T) ** 2 + (y_T / self.T) ** 2))
        L = torch.mean(torch.norm(ode, p=2) ** 2)
        self.ploss.append(L.item())
        return L

    def goal_loss(self, x, y, x_T, y_T, t):
        L = self.T
        self.gloss.append(L.item())
        return L

    def pinn_loss(self, x, y, t, loss_weights, loss_fns):
        self.x_T = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0].reshape((-1, 1)).to(device)
        self.y_T = grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0].reshape((-1, 1)).to(device)

        return sum([w * fn(x, y, self.x_T, self.y_T, t) for w, fn in zip(loss_weights, loss_fns)])

    def train(self, model, optimizer, steps):
        model.train()

        print("Adam...")
        for n in range(steps):
            u = model(self.time_domain)
            x, y = u[:, 0], u[:, 1]
            loss = self.pinn_loss(x, y, self.time_domain, self.weights, [self.constraint_loss, self.physics_loss, self.goal_loss])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if n % 100 == 0:
                self.time_domain = self.pde_resampler()

            if n % 1000 == 0:
                print(f"Step {n}, Loss {loss.item()}, T: {self.T.item()}")

        def closure():
            optimizer.zero_grad()
            u = model(self.time_domain)
            x, y = u[:, 0], u[:, 1]
            loss = self.pinn_loss(x, y, self.time_domain, self.weights, [self.constraint_loss, self.physics_loss, self.goal_loss])

            loss.backward()
            return loss

        print("L-BFGS...")
        optimizer = torch.optim.LBFGS(list(model.parameters()), lr=self.lr_lbfgs)
        try:
            for n in range(self.n_lbfgs):
                optimizer.step(closure)
                loss = closure()
                if n % 100 == 0:
                    self.time_domain = self.pde_resampler()
                print(f"Step {n}, Loss: {loss.item()}, T: {self.T.item()}")

                if (loss > 1e2 and n > 100) or loss < 0:
                    break
        except KeyboardInterrupt:
            pass

        torch.save({
            'epoch': n,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'data/model_{time.strftime("%Y%m%d-%H%M%S")}.pt')

    def plot(self, model):
        model.eval()
        with torch.no_grad():
            u = model(self.time_domain)
            x, y = u[:, 0], u[:, 1]
            print(x[0].item(), y[0].item())
            print(x[-1].item(), y[-1].item())
            plt.plot(x.detach().numpy(), y.detach().numpy(), label='pinn')

            r = 0.5729
            theta = np.linspace(0, np.arccos(1 - 1 / r), 1001)
            x = r * (theta - np.sin(theta))
            y = 1 - r * (1 - np.cos(theta))
            plt.plot(x, y, 'k--', label='analytic')
            plt.xlim((-0.1, 1.1))
            plt.ylim((-0.1, 1.1))
            plt.legend()
            plt.grid()
            plt.savefig(f'data/plot_{time.strftime("%Y%m%d-%H%M%S")}.png')
            plt.show()


if __name__ == "__main__":
    spb = SPBPINN()
    optimizer = torch.optim.Adam(list(spb.net.parameters()), lr=spb.lr_adam)
    spb.train(spb.net, optimizer, spb.n_adam)
    spb.plot(spb.net)

    # plt.semilogy(spb.ploss[500:], label='physics')
    plt.semilogy(spb.closs[500:], label='constraint')
    plt.semilogy(spb.gloss[500:], label='goal')
    plt.legend()
    plt.show()
