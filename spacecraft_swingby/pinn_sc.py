import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
import random
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class Net(nn.Module):
    """
    Forward neural network architecture for the PINN. Network consists of 1 input (time), 3 hidden layers with 64 neurons each,
    and 2 outputs (x and y positions of the spacecraft). A tanh activation function is used for all hidden layers and the output layer.
    """
    def __init__(self, layers):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.activation = torch.tanh

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return x


class PINN():
    """
    This class holds the PINN for the spacecraft swingby problem.
        :param bh_xygm: List of lists containing the x, y, and gravitational mass of the black holes.
        :param lr_adam: Learning rate for the Adam optimizer.
        :param lr_lbfgs: Learning rate for the L-BFGS optimizer.
        :param n_adam: Number of Adam iterations.
        :param n_lbfgs: Number of L-BFGS iterations.
        :param weights: Tuple of weights for the constraint and physics loss functions.

    To run and train the PINN, run the following commands:
        pinn = PINN()
        optimizer = torch.optim.Adam(list(pinn.net.parameters()) + [pinn.T], lr=pinn.lr_adam)
            -> [trains both the model parameters and the external T variable]
        pinn.train(pinn.net, optimizer, pinn.n_adam)
        pinn.plot(pinn.net)
    """
    def __init__(self, bh_xygm=None, lr_adam=4e-3, lr_lbfgs=3.8e-3, n_adam=2000, n_lbfgs=4000, weights=(10., 1.)):
        self.tmin, self.tmax = 0.0, 1.0
        self.x0, self.y0 = -1., -1.
        self.x1, self.y1 = 1., 1.
        self.bh_xygm = [[-0.5, -1.0, 0.5],
                        [-0.2, 0.4, 1.0],
                        [0.8, 0.3, 0.5],] if bh_xygm is None else bh_xygm

        self.n_output = 2
        self.n_adam = n_adam
        self.n_lbfgs = n_lbfgs
        self.n_domain = 1000
        self.lr_adam, self.lr_lbfgs = lr_adam, lr_lbfgs
        self.xT, self.xTT = None, None
        self.yT, self.yTT = None, None

        self.weights = weights

        self.net = Net([1] + [64] * 3 + [self.n_output])
        self.net.to(device)
        self.time_domain = torch.linspace(self.tmin, self.tmax, self.n_domain, requires_grad=True).reshape((self.n_domain, 1)).to(device)
        self.T = torch.autograd.Variable(torch.tensor([1], dtype=torch.float32).to(device), requires_grad=True)

    def pde_resampler(self):
        """
        Randomly resample the time domain for the spacecraft swingby problem. This is called every 100 epochs during training.
        """
        points = [self.tmin]
        points += [random.uniform(self.tmin, self.tmax) for _ in range(self.n_domain-2)]
        points += [self.tmax]
        points.sort()

        points_tensor = torch.tensor(points).reshape((self.n_domain, 1)).to(device)
        points_tensor.requires_grad = True

        return points_tensor

    def constraint_loss(self, x, y, x_T, x_TT, y_T, y_TT, t):
        return ((x[0] - self.x0) ** 2 + (y[0] - self.y0) ** 2 + (x[-1] - self.x1) ** 2 + (y[-1] - self.y1) ** 2)

    def physics_loss(self, x, y, x_T, x_TT, y_T, y_TT, t):
        ode_x = x_TT
        ode_y = y_TT
        for xtmp, ytmp, gmtmp in self.bh_xygm:
            ode_x += (gmtmp * (x.reshape((-1, 1)) - xtmp) /
                      ((x.reshape((-1, 1)) - xtmp) ** 2 + (y.reshape((-1, 1)) - ytmp) ** 2) ** 1.5)
            ode_y += (gmtmp * (y.reshape((-1, 1)) - ytmp) /
                      ((x.reshape((-1, 1)) - xtmp) ** 2 + (y.reshape((-1, 1)) - ytmp) ** 2) ** 1.5)

        return torch.mean(torch.norm(ode_x, p=2) ** 2) + torch.mean(torch.norm(ode_y, p=2) ** 2)

    def pinn_loss(self, x, y, t, loss_weights, loss_fns):
        """
        Defines the loss function for the PINN, as a weighted combination of the constraint and physics loss functions, to be minimized,
        hence enforcing the initial & final positions and the overall physics of the spacecraft swingby problem.
        """
        x_T = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0].reshape((-1, 1)).to(device) / self.T
        y_T = grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0].reshape((-1, 1)).to(device) / self.T
        self.x_TT = grad(x_T, t, grad_outputs=torch.ones_like(x_T), create_graph=True)[0].reshape((-1, 1)).to(device) / self.T
        self.y_TT = grad(y_T, t, grad_outputs=torch.ones_like(y_T), create_graph=True)[0].reshape((-1, 1)).to(device) / self.T

        return sum([w * fn(x, y, x_T, self.x_TT, y_T, self.y_TT, t) for w, fn in zip(loss_weights, loss_fns)])


    def train(self, model, optimizer, steps):
        """
        Training of the PINN, first by applying the Adam optimizer for n_adam epochs, then the L-BFGS optimizer for n_lbfgs epochs.
        Model checkpoint is saved at the end of training in the /data/ directory.
        """
        model.train()

        print("Adam...")
        for n in range(steps):
            u = model(self.time_domain)
            x, y = u[:, 0], u[:, 1]
            loss = self.pinn_loss(x,  y, self.time_domain, self.weights, [self.constraint_loss, self.physics_loss])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if n % 100 == 0:
                self.time_domain = self.pde_resampler()

            if n % 1000 == 0:
                print(f"Step {n}, Loss: {loss.item()}, T: {self.T.item()}")

        def closure():
            optimizer.zero_grad()
            u = model(self.time_domain)
            x, y = u[:, 0], u[:, 1]
            loss = self.pinn_loss(x, y, self.time_domain, self.weights, [self.constraint_loss, self.physics_loss])
            loss.backward()
            return loss

        print("L-BFGS...")
        optimizer = torch.optim.LBFGS(list(model.parameters()) + [self.T], lr=self.lr_lbfgs)
        try:
            for n in range(self.n_lbfgs):
                optimizer.step(closure)
                loss = closure()
                if n % 100 == 0:
                    self.time_domain = self.pde_resampler()
                print(f"Step {n}, Loss: {loss.item()}, T: {self.T.item()}")
        except KeyboardInterrupt:
            pass

        torch.save({
            'epoch': n,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'data/model_{time.strftime("%Y%m%d-%H%M%S")}.pt')


    def plot(self, model):
        """
        Plots trajectory of the spacecraft with respect to the other astronomical bodies in the system.
        Plots saved in the /data/ directory.
        """
        model.eval()
        with torch.no_grad():
            u = [model(t.reshape(1)).to(device) for t in self.time_domain]
        x = torch.stack([u[i][0] for i in range(self.n_domain)]).cpu()
        y = torch.stack([u[i][1] for i in range(self.n_domain)]).cpu()
        plt.figure(figsize=(6, 6))
        plt.plot(x.detach().numpy(), y.detach().numpy())
        for i, (xtmp, ytmp, gmtmp) in enumerate(self.bh_xygm):
            plt.scatter(xtmp, ytmp, s=gmtmp * 500, c='r', marker='o')
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.1, 1.1))
        plt.grid()
        plt.savefig(f'data/plot_{time.strftime("%Y%m%d-%H%M%S")}.png', dpi=300)
        plt.show()


    def retrain(self, model, optimizer, steps):
        """
        Can optionally retrain the model with a different optimizer and number of epochs.
        Useful to further reduce the loss function and improve the accuracy of the model.
        Model checkpoint is saved at the end of training in the /data/ directory.
        """
        model.train()

        def closure():
            optimizer.zero_grad()
            u = model(self.time_domain)
            x, y = u[:, 0], u[:, 1]
            loss = self.pinn_loss(x, y, self.time_domain, loss_weights=self.weights, loss_fns=[self.constraint_loss, self.physics_loss])
            loss.backward()
            return loss

        print("L-BFGS part 2 ...")
        try:
            for n in range(steps):
                optimizer.step(closure)
                loss = closure()
                if n % 100 == 0:
                    self.time_domain = self.pde_resampler()
                print(f"Step {n}, Loss: {loss.item()}, T: {self.T.item()}")
        except KeyboardInterrupt:
            pass

        torch.save({
            'epoch': n,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'data/model_{time.strftime("%Y%m%d-%H%M%S")}.pt')


if __name__ == '__main__':
    pinn = PINN()
    optimizer = torch.optim.Adam(list(pinn.net.parameters()) + [pinn.T], lr=pinn.lr_adam)
    pinn.train(pinn.net, optimizer, pinn.n_adam)
    pinn.plot(pinn.net)
    optimizer = torch.optim.LBFGS(list(pinn.net.parameters()) + [pinn.T], lr=0.05)
    pinn.retrain(pinn.net, optimizer, 4000)
    pinn.plot(pinn.net)

    # file1 = 'data/model_20240328-141516.pt'
    # checkpoint = torch.load(file1)
    # pinn.net.load_state_dict(checkpoint['model_state_dict'])
    # pinn.T = torch.autograd.Variable(torch.tensor([1.9], dtype=torch.float32).to(device), requires_grad=True)
    # optimizer = torch.optim.LBFGS(list(pinn.net.parameters()) + [pinn.T], lr=0.05)
    # pinn.retrain(pinn.net, optimizer, 4000)
    # pinn.plot(pinn.net)
    # with open('data/T.txt', 'w') as f:
    #     f.write(f"{pinn.T.item()}\n")

    # file1 = 'data/model_20240328-182213.pt'
    # checkpoint = torch.load(file1)
    # pinn.net.load_state_dict(checkpoint['model_state_dict'])
    # pinn.T = torch.autograd.Variable(torch.tensor([2.0988], dtype=torch.float32).to(device), requires_grad=True)
    # optimizer = torch.optim.LBFGS(list(pinn.net.parameters()) + [pinn.T], lr=0.07)
    # pinn.retrain(pinn.net, optimizer, 4000)
    # pinn.plot(pinn.net)
    # with open('data/T.txt', 'w') as f:
    #     f.write(f"{pinn.T.item()}\n")

