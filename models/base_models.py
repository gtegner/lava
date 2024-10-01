import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.cnn_base_models import ConvBase
from models.cnn_base_models import CNNFilm


def param_to_vec(list_of_params):
    num_params = sum([np.prod(x.shape) for x in list_of_params])
    vec = torch.zeros(num_params)
    index = 0
    for param in list_of_params:
        param_flat = param.reshape(-1)
        vec[index : index + len(param_flat)] = param_flat
        index += len(param_flat)

    return vec

def vec_to_param(vec, theta_shapes):
    index = 0
    param_list = []
    for shape in theta_shapes:
        n = np.prod(shape)
        vec_sub = vec[index : index + n]
        param_list.append(vec_sub.reshape(shape))
        index += n
    return param_list

class VectorField(nn.Module):
    def __init__(self, input_dim, output_dim, adaptation, context_dim, feature_extractor='nn', objective='regression'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_extractor = feature_extractor

        self.main_net = MainNetwork(input_dim, output_dim, context_dim, adaptation, feature_extractor=feature_extractor)
        self.num_params = self.main_net.num_params
        self.objective = objective

    def forward(self, x, z):
        return self.main_net(x, z)

    @property
    def initialization(self):
        return self.main_net.initialization

    def loss_fn(self, x, y, z):
        y_pred = self.forward(x, z)
        return self.criterion(y_pred, y)

    def criterion(self, y_pred, y):
        if self.objective == 'regression':
            return F.mse_loss(y_pred, y)
        elif self.objective == 'classification':
            return F.cross_entropy(y_pred, y)

    def optimize(self, opt, loss, reg_loss=0):
        self.main_net.optimize(opt, loss, reg_loss=reg_loss)

class ConditionalNet(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim):
        super().__init__()
        hidden = 64
        self.net = nn.Sequential(
            nn.Linear(input_dim + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
        
    def forward(self, x, z):
        if len(x.shape) == 1 and len(z.shape) == 1:
            xz = torch.cat((x, z))
        else:
            z = z.unsqueeze(0).repeat(x.shape[0], 1)
            xz = torch.cat((x,z), -1)
        return self.net(xz)


class LastLayerNet(nn.Module):
    def __init__(self, input_dim, output_dim, feature_extractor='nn'):
        super().__init__()

        hidden = 64

        if feature_extractor == 'nn':
            self.base_network = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            latent_dim = hidden

        elif feature_extractor == 'conv':
            #h_shape = 5
            #filters = 64
            self.base_network = ConvBase(output_size=hidden, nc=input_dim)
            #latent_dim = h_shape * h_shape * filters
            latent_dim = 128


        output_layer = nn.Linear(latent_dim, output_dim)
        self.theta_shapes = [x.shape for x in output_layer.parameters()]
        self.num_params = sum([np.prod(s) for s in self.theta_shapes])

        self.output_dim = output_dim
        self.hidden = hidden

        self.initialization = param_to_vec(list(output_layer.parameters()))


    def forward(self, x, theta):
        h = self.base_network(x)
        return F.linear(h, weight=theta[0], bias=theta[1])



class FullNet(nn.Module):
    def __init__(self, input_dim, output_dim, feature_extractor):
        super().__init__()

        hidden = 64
        self.feature_extractor = feature_extractor

        if self.feature_extractor == 'nn':
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            )
        elif self.feature_extractor == 'conv':
            self.net = nn.Sequential(ConvBase(output_dim, 3, 128), nn.Linear(128, output_dim))

        self.theta0 = list(self.net.parameters())
        self.theta_shapes = [x.shape for x in self.theta0]
        self.num_params = np.sum([np.prod(x) for x in self.theta_shapes])
        self.latent_dim = hidden


    def forward(self, x, theta0=None):
        if theta0 is None:
            theta0 = self.theta0

        if self.feature_extractor == 'nn':
            n = len(theta0)
            h = x
            for i in range(0, n - 2, 2):
                h = F.relu(F.linear(h, theta0[i], bias=theta0[i+1]))
            return F.linear(h, theta0[-2], bias=theta0[-1])
        elif self.feature_extractor == 'conv':
            h = x
            theta=theta0
            for i in range(0, 4*4, 4):
                conv = F.conv2d(h, weight=theta[i], bias=theta[i+1], padding=1)
                h = F.max_pool2d(F.relu(F.batch_norm(conv, running_mean=None, running_var=None, weight=theta[i+2], bias=theta[i+3], training=True, momentum=1.0)), 2)
            h = torch.flatten(h, start_dim=1)
            h = F.linear(h, weight=theta[-4], bias=theta[-3])
            h = F.linear(h, weight=theta[-2], bias=theta[-1])

            return h




class MainNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim, adaptation, feature_extractor='nn'):
        super().__init__()

        self.adaptation = adaptation

        self.initialization_ = None
        if adaptation == 'head':
            self.net = LastLayerNet(input_dim, output_dim, feature_extractor)
            self.initialization_ = self.net.initialization
            self.num_params = self.net.num_params

        elif adaptation == 'full':
            self.net = FullNet(input_dim, output_dim, feature_extractor)
            self.initialization_ = self.net.theta0
            self.num_params = self.net.num_params

        elif adaptation == 'conditional':

            if feature_extractor == 'nn':
                self.net = ConditionalNet(input_dim, output_dim, context_dim)

            elif feature_extractor == 'conv':
                self.net = CNNFilm(input_dim, context_dim, 64, output_dim, n_layers=4, use_batchnorm=True)

            self.num_params = context_dim
        else:
            raise NotImplementedError(f"Adaptation: {adaptation} not implemented")

    @property
    def initialization(self):
        if self.initialization_ is not None:
            return self.initialization_
        return torch.randn(self.num_params)

    def forward(self, x, theta):
        if self.adaptation == 'head':
            theta = vec_to_param(theta, self.net.theta_shapes)
        return self.net(x, theta)

    def optimize(self, opt, loss, xq=None, yq=None, reg_loss=0):
        opt.zero_grad()
        (loss + reg_loss).backward()
        opt.step()


