import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def conv3x3(in_channels, out_channels, **kwargs):
    # The convolutional layers (for feature extraction) use standard layers from
    # `torch.nn`, since they do not require adaptation.
    # See `examples/maml/model.py` for comparison.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvBase(nn.Module):
    def __init__(self, output_size, nc, hidden=64):
        super().__init__()

        filters = hidden
        filters = 64

        self.features = nn.Sequential(
            conv3x3(nc, filters),
            conv3x3(filters, filters),
            conv3x3(filters, filters),
            conv3x3(filters, filters),
        )

        self.fc_out = nn.Linear(5 * 5 * filters, 128)

    def forward(self, x):
        h = self.features(x)
        h_flat = h.view(h.shape[0], -1)

        return self.fc_out(h_flat)


class CNNFilm(nn.Module):

    def __init__(self, input_size, z_size, hidden, output_size, n_layers, use_batchnorm):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.n_layers = n_layers
        self.z_size = z_size

        filters = hidden

        #h_shape = 1 if input_size[0] == 28 else 5
        h_shape = 5



        self.f1 = nn.ModuleList()
        self.f1.append(nn.Conv2d(input_size, filters, 2, stride=1, padding=1))
        self.f1.append(nn.Conv2d(filters, filters, 2, stride=1, padding=1))
        self.f1.append(nn.Conv2d(filters, filters, 2, stride=1, padding=1))

        self.bn1 = nn.BatchNorm2d(filters, track_running_stats=False) if self.use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(filters, track_running_stats=False) if self.use_batchnorm else nn.Identity()
        self.bn3 = nn.BatchNorm2d(filters, track_running_stats=False) if self.use_batchnorm else nn.Identity()
        self.bn4 = nn.BatchNorm2d(filters, track_running_stats=False) if self.use_batchnorm else nn.Identity()

        self.f2 = nn.ModuleList()
        self.f2.append(nn.Conv2d(filters, filters, 3, stride=1, padding=1))
        self.f2.append(nn.Flatten())
        self.f2.append(nn.Linear(h_shape * h_shape * filters, output_size))

        self.MP = nn.MaxPool2d(2)

        self.film_layer = nn.Linear(z_size, filters*2)
        self.theta_shapes = []
        for layer in self.f1:
            for param in layer.parameters():
                self.theta_shapes.append(list(param.shape))
        for layer in self.f2:
            for param in layer.parameters():
                self.theta_shapes.append(list(param.shape))

        for m in self.f1:
            self.init_weights(m)

        for m in self.f2:
            self.init_weights(m)

        torch.nn.init.kaiming_uniform_(self.film_layer.weight, nonlinearity='linear')



    def get_theta_shape(self):
        return self.theta_shapes

    def get_theta_size_per_layer(self):
        return [np.prod(shape) for shape in self.theta_shapes]

    def get_param_list(self, theta):
        layer_size = self.get_theta_size_per_layer()
        return [theta[0, int(np.sum(layer_size[:i])):int(np.sum(layer_size[:i]))+layer_size[i]].view(self.theta_shapes[i]) for i, s in enumerate(layer_size)]

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')

            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    def forward(self, x, theta, l_start=0, l_finish=-1):

        if self.z_size > 0:

            if x.shape[0] != theta.shape[0]:
                theta = theta.repeat(x.shape[0], 1)

            z = self.film_layer(theta)
            beta = torch.unsqueeze(torch.unsqueeze(z[:, :int(z.shape[-1] / 2)], -1), -1)
            gamma = torch.unsqueeze(torch.unsqueeze(z[:, int(z.shape[-1] / 2):], -1), -1)

            h = F.relu(self.MP(self.bn1(self.f1[0](x))))

            h = F.relu(self.MP(self.bn2(self.f1[1](h))))

            h = self.MP(self.bn3(self.f1[2](h)))

            h = F.relu(gamma * h + beta)

            h = F.relu(self.MP(self.bn4(self.f2[0](h))))

            h = self.f2[1](h)
            out = self.f2[2](h)

            return out

        h = self.MP(F.relu(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=1)))
        h = self.MP(F.relu(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=1)))
        h = self.MP(F.relu(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=1)))
        h = self.MP(F.relu(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=1)))
        h = torch.flatten(h, start_dim=1)
        out = F.linear(h, theta[8], bias=theta[9])

        return out
