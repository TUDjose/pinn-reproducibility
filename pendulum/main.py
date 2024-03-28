import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad
import random

class FNN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)
        ])
        self.activation = F.tanh

        # initialize (Glorot normal)
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            #nn.init.xavier_normal_(layer.bias)

    def forward(self, x):
        for layer in self.linears[:-2]:
            x = self.activation(layer(x))
            
        return self.linears[-1](x)
    
class PendulumPINN(FNN):
    def __init__(self):
        # Time domain input
        # Theta, Torque output
        super().__init__([1] + [64] * 3 + [2])
    
    def forward(self, x):
        x = super().forward(x)
        # theta, torque = x.T
        # x = torch.stack([theta, F.tanh(torque) * max_torq]).T
        return x

# hyperparameters

# Two outputs:
# theta: angle from rest
# torque: torque of logit
n_output = 2 # theta, torq_norm

# theta, theta_t, torque
initial_conditions = [0, 0, 0]

max_torq = 1.5
tmin, tmax = 0.0, 10.0
m = 1
l = 1
g = 9.8

n_adam = 5000
n_domain = 1000
device = "cpu"

theta_t = None
theta_tt = None
losses = []


def pinn_loss(theta, torq, t, loss_weights=[10, 1, 1], save_losses=True):
    # Calculate derivatives
    global theta_t
    global theta_tt
    global losses
    theta_t = grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0].reshape((1, -1))
    theta_tt = grad(theta_t, t, grad_outputs=torch.ones_like(theta_t), create_graph=True)[0].reshape((1, -1))

    torq = torch.tanh(torq) * max_torq

    # Initial conditions
    L_con = 0
    if t[0] == tmin:
        L_con = (theta[0])**2 + (torq[0])**2 + (theta_t[0][0])**2 


    # Physics loss
    gravity = m * l * l * theta_tt - (torq - m * g * l * torch.sin(theta))
    # calculate squared L2 norm
    L_phys = gravity.abs().square().mean()

    # Goal loss
    if t[-1] == tmax:
        L_goal = (torch.cos(theta[-1]) - (-1))**2

    L_con = loss_weights[0] * L_con
    L_phys = loss_weights[1] * L_phys
    L_goal = loss_weights[2] * L_goal

    if save_losses:
        losses.append((L_con, L_phys, L_goal))

    return L_con + L_phys + L_goal
    

# net = FNN([1] + [64] * 3 + [n_output])
net = PendulumPINN()

time_domain = torch.linspace(tmin, tmax, n_domain)
time_domain = time_domain.reshape((n_domain, 1))
time_domain.requires_grad = True

def sample_points(tmin, tmax):
    points = [tmin]
    points += [random.uniform(tmin, tmax) for _ in range(n_domain-2)]
    points += [tmax]
    points.sort()
    
    # Convert the list to a PyTorch tensor
    points_tensor = torch.tensor(points).reshape((n_domain, 1))
    points_tensor.requires_grad = True

    return points_tensor

def train(model, optimizer, steps):
    # set model to training mode
    model.train()

    loss_weights=[10, 1, 1]

    period = 100

    time_domain = sample_points(tmin, tmax)

    print("Adam..")
    for n in range(steps):
        # Forward
        optimizer.zero_grad()

        # Resample
        if n % period == 0:
            time_domain = sample_points(tmin, tmax)

        u = model(time_domain)
        theta, torq = u[:, 0], u[:, 1]
        loss = pinn_loss(theta, torq, time_domain, loss_weights=loss_weights)

        # Backprop
        loss.backward()
        optimizer.step()
        if n % 100 == 0:
            print(loss)

    # Define closure for L-BFGS
    def closure(time_domain, save_losses=False):
        optimizer.zero_grad()        
        u = model(time_domain)
        theta, torq = u[:, 0], u[:, 1]
        loss = pinn_loss(theta, torq, time_domain, loss_weights=loss_weights, save_losses=save_losses)
        loss.backward()
        return loss
    
    print("L-BFGS..")
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
    time_domain = torch.linspace(tmin, tmax, n_domain)
    time_domain = time_domain.reshape((n_domain, 1))
    time_domain.requires_grad = True

    try:
        for n in range(steps):
            # if n % period == 0:
            #     time_domain = sample_points(tmin, tmax)
            optimizer.step(lambda: closure(time_domain))
            loss = closure(time_domain, save_losses=False)
            print(loss)
    except KeyboardInterrupt:
        print("Interrupted")
        pass

def plot_output(model):
    model.eval()

    with torch.no_grad():
        u = [model(t.reshape(1)).to(device=device) for t in time_domain]
    

    plt.subplot(1, 2, 1)  
    plt.plot(time_domain.detach().numpy(), [y[0] for y in u], label="theta")
    plt.plot(time_domain.detach().numpy(), theta_t.reshape((-1, 1)).detach().numpy(), label="theta_t", alpha=0.5)
    plt.plot(time_domain.detach().numpy(), theta_tt.reshape((-1, 1)).detach().numpy(), label="theta_tt", alpha=0.5)

    # plt.plot(time_domain.detach().numpy(), [torch.tanh(y[1]) * max_torq for y in u], label="torque")
    plt.plot(time_domain.detach().numpy(), [torch.tanh(y[1]) for y in u], label="torque")
    plt.legend()

    # Plot losses
    plt.subplot(1, 2, 2)  
    plt.plot([x[0].detach().numpy() for x in losses], label="constraint")
    plt.plot([x[1].detach().numpy() for x in losses], label="physics")
    plt.plot([x[2].detach().numpy() for x in losses], label="goal")
    plt.semilogy()
    plt.legend()

    plt.show()


time_domain = time_domain.to(device=device)
net.to(device=device)
print("Training..")

train(net, torch.optim.Adam(net.parameters(), lr=2e-2), n_adam)

print("Plotting..")

plot_output(net)

