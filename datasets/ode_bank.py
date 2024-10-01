import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint


def Constant(x):
    return torch.ones_like(x)

#def SineFreq(x, freq):
    #return 


class ODE(nn.Module):
    def __init__(self, dt, steps):
        super().__init__()
        self.params = None
        self.t0 = 0
        self.t_end = steps * dt
        self.t_range = torch.linspace(self.t0, self.t_end, steps)

    def get_t_range(self, param=None):
        return self.t_range

    def vector_field(self, x, t, param):
        # x \ in [batch x dim]
        raise NotImplementedError

    def solve(self, initial, t, param):
        return odeint(lambda t, x : self.vector_field(x, t, param), initial, t)

    def sample_from_domain(self, *shape, param=None):
        # Always a square domain
        return np.random.uniform(self.x_min, self.x_max, size=shape + (self.dim, ))

    def __call__(self, xy, t, params):
        return self.vector_field(xy, t, params)

    def generate_params(self, num_params):
        raise NotImplementedError

    def generate_params(self, size):
        return np.random.uniform(-4, 4, size=(size, 1, self.param_dim))

    def viz(self):

        pass


class Sine(ODE):
    def __init__(self, dt, steps):
        super().__init__(dt, steps)

        self.name = 'sine'

        self.dim = 1
        self.x_min = -5
        self.x_max = 5
        self.param_dim = 2
        
    def vector_field(self, xy, t, params):

        z1 = params[0] * torch.sin(xy + params[1])
        return z1

    def generate_params(self, size):
        A = np.random.uniform(0.1, 5.0, size=(size, 1))
        phi = np.random.uniform(-np.pi, np.pi, size=(size, 1))
        return np.concatenate((A, phi), -1)

    def __call__(self, xy, t, params):
        return self.vector_field(xy, t, params)

class PendulumODE(ODE):
    def __init__(self, dt, steps):
        super().__init__(dt, steps)

        self.dim = 2
        self.param_dim = 3
        self.name = 'pendulum'

        self.theta_min = -np.pi/2
        self.theta_max = np.pi/2

        self.p_min = -1
        self.p_max = 1
        #self.std = self.get_vf_std()

    def sample_from_domain(self, *shape, param=None):
        # Always a square domain
        n = shape[0]
        theta_samples = np.random.uniform(self.theta_min, self.theta_max, size=(*shape, 1))
        vel_samples = np.random.uniform(self.p_min, self.p_max, size=(*shape, 1))
        samples = np.concatenate((theta_samples, vel_samples), -1)
        return samples
        

    def vector_field(self, xy, t, params):
        m, g, l = params
        theta = xy[..., 0:1]
        p_theta = xy[..., 1:2]

        z1 = p_theta / (m * l**2)
        z2 = -m * g * l * torch.sin(theta)

        return torch.cat((z1, z2), -1)

    def generate_params(self, size, ood=False):
        if ood:
            m = np.random.uniform(1.5, 2.5, size=(size, 1))
            g = np.random.uniform(0.5, 1.5, size=(size, 1))
            l = np.random.uniform(1.5, 2.0, size=(size, 1))

        else:
            m = np.random.uniform(0.5, 1.5, size=(size, 1))
            g = np.random.uniform(0.5, 1.5, size=(size, 1))
            l = np.random.uniform(0.5, 1.5, size=(size, 1))
        return np.concatenate((m,g,l), -1)

    def __call__(self, xy, t, params):
        return self.vector_field(xy, t, params)


class VanDerPolOscillator(ODE):
    def __init__(self, dt, steps):
        super().__init__(dt, steps)

        self.name = 'vanderpol'
        self.dim = 2
        self.x_min = -3
        self.x_max = 3
        self.param_dim = 1

    def vector_field(self, xy, t, params):
        x, y = xy[..., 0:1], xy[..., 1:2]
        mu = params[0]

        dx = y
        dy = mu * (1 - x**2) * y - x

        return torch.cat((dx, dy), -1)

    def generate_params(self, size):
        return np.random.uniform(0.1, 5, size=(size, 1))

    def __call__(self, xy, t, params):
        return self.vector_field(xy, t, params)

class FitzHughNagumo(ODE):
    def __init__(self, dt, steps):
        super().__init__(dt, steps)

        self.name = 'fitzhugh-nagumo'
        self.dim = 2
        self.x_min = -2.5
        self.x_max = 2.5
        self.param_dim = 3

    def vector_field(self, uv, t, params):
        u, v = uv[..., 0:1], uv[..., 1:2]
        a, b, c = params[0], params[1], params[2]

        du = c * (u - u**3 / 3 + v)
        dv = -1 / c * (u - a + b * v)

        return torch.cat((du, dv), -1)

    def generate_params(self, size):
        return np.random.uniform(0.1, 2, size=(size, 3))

    def __call__(self, uv, t, params):
        return self.vector_field(uv, t, params)


class MassSpringODE(ODE):
    def __init__(self, dt, steps):
        super().__init__(dt, steps)

        self.dim = 2
        self.x_min = -1
        self.x_max = 1
        self.param_dim = 2
        self.name = 'spring'
        

    def vector_field(self, xy, t, params):
        m, k = params
        x = xy[..., 0:1]
        p_x = xy[..., 1:2]

        z1 = p_x / m
        z2 = -k * x

        return torch.cat((z1, z2), -1)

    def generate_params(self, n, ood=False):
        if ood:
            mass = np.random.uniform(1.5, 2.5, size=(n, 1))
            k = np.random.uniform(1.5, 2.5, size=(n, 1))
        else:
            mass = np.random.uniform(0.5, 1.5, size=(n, 1))
            k = np.random.uniform(0.5, 1.5, size=(n, 1))
        return np.concatenate((mass, k), -1)

    def __call__(self, xy, t, params):
        return self.vector_field(xy, t, params)


if __name__ == '__main__':
    pass