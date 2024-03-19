import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad

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

def constraint_loss(theta, theta_t, theta_tt, torq, t):
    # need to add constraint for theta_t[0] = 0 but it doesn't seem to work very well
    # have also set the goal to 0 for now
    # print(theta, theta_t)
    return 10*(theta[0] - torch.pi/3)**2 + torch.mean(torch.norm(torq, p=2)**2) + (theta_t[0][0])**2 

theta_t = None
theta_tt = None
def physics_loss(theta, theta_t, theta_tt, torq, t):
    # calculate second derivative somehow
    
   
    torq = F.tanh(torq) * max_torq
    gravity = m * l * l * theta_tt - (torq - m * g * l * torch.sin(theta))
    # calculate squared L2 norm
    x = torch.mean(torch.norm(gravity, p=2)**2)
    return x

def goal_loss(theta, theta_t, theta_tt, torq, t, target_time=tmax):
    return torch.square((torch.cos(theta) * int(t == target_time) + 1))


def pinn_loss(theta, torq, t, loss_weights, loss_fns):
    global theta_t
    global theta_tt
    theta_t = grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0].reshape((1, -1))
    theta_tt = grad(theta_t, t, grad_outputs=torch.ones_like(theta_t), create_graph=True)[0].reshape((1, -1))
    return sum([w * L(theta, theta_t, theta_tt, torq, t) for w, L in zip(loss_weights, loss_fns)])
    

net = FNN([1] + [64] * 3 + [n_output])

time_domain = torch.linspace(tmin, tmax, n_domain)
time_domain = time_domain.reshape((n_domain, 1))
time_domain.requires_grad = True

def train(model, optimizer, steps):
    # set model to training mode
    model.train()

    print("Adam..")
    for n in range(steps):
        # Forward
        u = model(time_domain)
        theta, torq = u[:, 0], u[:, 1]
        loss = pinn_loss(theta, torq, time_domain, loss_weights=[1, 1], loss_fns=[constraint_loss, physics_loss])

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if n % 100 == 0:
            print(loss)

    # Define closure for L-BFGS
    def closure():
        optimizer.zero_grad()
        u = model(time_domain)
        theta, torq = u[:, 0], u[:, 1]
        loss = pinn_loss(theta, torq, time_domain, loss_weights=[1, 1], loss_fns=[constraint_loss, physics_loss])
        loss.backward()
        return loss
    
    print("L-BFGS..")
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-1)
    try:
        for n in range(steps):
            optimizer.step(closure)
            loss = closure()
            print(loss)
    except KeyboardInterrupt:
        pass

def plot_output(model):
    model.eval()

    with torch.no_grad():
        u = [model(t.reshape(1)).to(device=device) for t in time_domain]
    
    print(u)
    plt.plot(time_domain.detach().numpy(), [y[0] for y in u], label="theta")
    plt.plot(time_domain.detach().numpy(), theta_t.reshape((-1, 1)).detach().numpy(), label="theta_t")
    plt.plot(time_domain.detach().numpy(), theta_tt.reshape((-1, 1)).detach().numpy(), label="theta_tt")

    plt.plot(time_domain.detach().numpy(), [torch.tanh(y[1]) * max_torq for y in u], label="torque")
    plt.legend()
    plt.show()


time_domain = time_domain.to(device=device)
net.to(device=device)
print("Training..")

train(net, torch.optim.Adam(net.parameters(), lr=2e-3), n_adam)

print("Plotting..")

plot_output(net)


