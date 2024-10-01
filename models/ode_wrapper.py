import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vector_field = model
        self.num_params = self.vector_field.num_params
        self.adaptation = self.vector_field.adaptation

        self.input_dim = 1
        self.output_dim = self.vector_field.output_dim

    def loss_fn(self, t, y, z):
        y0 = y[0].unsqueeze(0)
        y_traj = self.simulate(y0, t, z).squeeze(1)
        loss = self.vector_field.criterion(y_traj, y)
        return loss

    def simulate(self, initial, times, theta):
        return odeint(lambda t, x : self.vector_field(x, theta), initial, times)

    def forward(self, initial, times, theta):
        if len(initial.shape) == 1:
            initial = initial.unsqueeze(0)
        out = self.simulate(initial, times, theta).squeeze(1)
        return out

    def criterion(self, ypred, y):
        return self.vector_field.criterion(ypred, y)


