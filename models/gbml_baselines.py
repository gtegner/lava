from models.base_models import MainNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_models import LastLayerNet
from torch.autograd.functional import hessian as f_hessian
import copy
import numpy as np
 


class MAMLOptimizer(nn.Module):

    def __init__(self, model, steps, det_reg=False):
        super().__init__()
        self.model = model

        if not isinstance(self.model.initialization, list):
            initialization = self.model.initialization if self.model.initialization is not None else torch.randn(self.model.num_params)
            self.theta0 = nn.Parameter(initialization)
        else:
            self.theta0 = self.model.initialization

        self.steps = steps
        self.learning_rate = nn.Parameter(torch.Tensor([0.1]))
        self.det_reg = det_reg

    def adapt_list(self, x, y, steps=None):
        if steps is None:
            steps = self.steps

        z = self.theta0
        for i in range(steps):
            loss = self.model.loss_fn(x, y, z)
            grads = torch.autograd.grad(loss, z, create_graph=True)
            z = list(map(lambda p: p[1] - self.learning_rate * p[0], zip(grads, z)))

        return z


    def adapt(self, x, y, steps=None):
        if isinstance(self.theta0, list):
            return self.adapt_list(x, y, steps)

        if steps is None:
            steps = self.steps

        z = self.theta0
        for i in range(steps):
            loss = self.model.loss_fn(x, y, z)
            grads = torch.autograd.grad(loss, z, create_graph=True)[0]
            z = z - self.learning_rate * grads
        return z

    def get_H_reg(self, x, y, theta):
        H = f_hessian(self.model.loss_fn, (x, y, theta), create_graph=True)[-1][-1]
        return 1e-6 * torch.log(torch.clamp(torch.linalg.det(H), min=1e-40))

    def forward(self, *args):
        return self.model(*args)

    def criterion(self, ypred, y):
        return self.model.criterion(ypred, y)

    def optimize(self, opt, loss, xq=None, yq=None, reg_loss=0):
        self.model.optimize(opt, loss, reg_loss=reg_loss)


class VR_MAML(nn.Module):

    def __init__(self, model, steps):
        super().__init__()
        self.model = model
        initialization = self.model.initialization if self.model.initialization is not None else torch.randn(self.model.num_params)

        self.theta0 = None
        self.theta1 = nn.ParameterList(initialization)
        self.copy_params()
        self.old_grads = None

        self.steps = steps
        self.learning_rate = nn.Parameter(torch.Tensor([0.001]))
        self.outer_learning_rate = 0.01
        self.gamma = 0.99


    def copy_params(self):
        self.theta0 = nn.ParameterList([copy.deepcopy(param) for param in self.theta1])

    def adapt(self, x, y, steps=None):
        if steps is None:
            steps = self.steps

        adapted_thetas = []
        for z in [self.theta0, self.theta1]:
            for i in range(steps):
                loss = self.model.loss_fn(x, y, z)
                grads = torch.autograd.grad(loss, z, create_graph=True)
                z = list(map(lambda p: p[1] - self.learning_rate * p[0], zip(grads, z)))
            adapted_thetas.append(z)

        return adapted_thetas

    def forward(self, *args):
        other_args = args[:-1]
        y1 = self.model(*(other_args + (args[-1][0],)))
        y2 = self.model(*(other_args + (args[-1][1],)))
        return [y1, y2]

    def criterion(self, ypred, y):
        return [self.model.criterion(ypred[0], y), self.model.criterion(ypred[1], y)]

    def optimize(self, opt, loss, xq=None, yq=None, reg_loss=0):

        l_old, l_new = loss[0], loss[1]
        ct1 = self.old_grads

        ct = torch.autograd.grad(outputs=l_new, inputs=self.theta1, create_graph=False)
        dt1 = torch.autograd.grad(outputs=l_old, inputs=self.theta0, create_graph=False)

        if ct1 is not None:
            ct_new = [c + (1 - self.gamma) * (c1 - d1) for c, c1, d1 in zip(ct, ct1, dt1)]
        else:
            ct_new = list(ct)

        self.copy_params()
        self.theta1 = nn.ParameterList([w - self.outer_learning_rate * c for w, c in zip(self.theta1, ct_new)])
        self.old_grads = ct_new


class VFML(MAMLOptimizer):

    def __init__(self, model, steps, det_reg=False):
        super().__init__(model, steps, det_reg=det_reg)

        if isinstance(self.theta0, list):
            self.v = [w * 0 for w in self.theta0]
        else:
            self.v = self.theta0 * 0

        self.gamma = 0.9
        self.beta = 0.5

    def adapt_list(self, x, y, steps=None):
        if steps is None:
            steps = self.steps

        z = self.theta0
        for i in range(steps):
            loss = self.model.loss_fn(x, y, z)
            grads = torch.autograd.grad(loss, z, create_graph=True)
            z = list(map(lambda p: p[1] - self.learning_rate * (p[0] + (1 - self.gamma) * p[2].to(p[0].device).detach()), zip(grads, z, self.v)))

        return z

    def adapt(self, x, y, steps=None):
        if isinstance(self.theta0, list):
            return self.adapt_list(x, y, steps)

        if steps is None:
            steps = self.steps

        z = self.theta0
        for i in range(steps):
            loss = self.model.loss_fn(x, y, z)
            grads = torch.autograd.grad(loss, z, create_graph=True)[0]
            z = z - self.learning_rate * (grads + (1 - self.gamma) * self.v.to(z.device).detach())
        return z

    def optimize(self, opt, loss, xq=None, yq=None, reg_loss=0):

        if isinstance(self.theta0, list):
            theta_0_t = [w.clone() for w in self.theta0]
        else:
            theta_0_t = self.theta0.clone()
        self.model.optimize(opt, loss, reg_loss=reg_loss)
        if isinstance(self.theta0, list):
            theta_0_t1 = [w.clone() for w in self.theta0]
        else:
            theta_0_t1 = self.theta0.clone()

        lt, lt1 = 0, 0
        xq = xq.to(loss.device)
        yq = yq.to(loss.device)
        for i in range(xq.shape[0]):
            lt += self.model.loss_fn(xq[i], yq[i], theta_0_t)
            lt1 += self.model.loss_fn(xq[i], yq[i], theta_0_t1)
        grad_t = torch.autograd.grad(outputs=(lt / xq.shape[0]), inputs=theta_0_t, allow_unused=True)
        grad_t1 = torch.autograd.grad(outputs=(lt1 / xq.shape[0]), inputs=theta_0_t1, allow_unused=True)

        if isinstance(self.v, list):
            self.v = [(g1 + (1 - self.beta) * (v.to(g.device) - g)).detach() for g, g1, v in zip(grad_t, grad_t1, self.v)]
        else:
            self.v = (grad_t1[0] + (1 - self.beta) * (self.v.to(grad_t[0].device) - grad_t[0])).detach()

class METAMIX(MAMLOptimizer):

    def __init__(self, model, steps, det_reg=False):
        super().__init__(model, steps, det_reg=det_reg)
        self.metamix = True

    def mix(self, xs, ys, xq, yq, theta):

        s_idx = np.random.choice(xs.shape[0], xq.shape[0])
        max_l = int(len(theta) / 2)
        l = np.random.randint(0, max_l)
        alpha = 0.5
        beta = 0.5
        lamb = torch.from_numpy(np.random.beta(alpha, beta, xq.shape[0])).view(-1, 1).float().to(xs.device)
        # if theta[0].shape[1] != xq.shape[1]:
        #     h_s = theta.unsqueeze(0).repeat(xs.shape[0], 1)
        #     h_s = torch.cat((xs, h_s), -1)
        #     h_q = theta.unsqueeze(0).repeat(xq.shape[0], 1)
        #     h_q = torch.cat((xq, h_q), -1)
        # else:
        h_s, h_q = xs, xq

        if self.model.feature_extractor == 'nn':
            for i in range(0, l * 2, 2):
                h_s = F.linear(h_s, theta[i], bias=theta[i + 1])
                h_q = F.linear(h_q, theta[i], bias=theta[i + 1])
                h_s = F.relu(h_s) if i != (len(theta) - 2) else h_s
                h_q = F.relu(h_q) if i != (len(theta) - 2) else h_q

            x_mix = lamb * h_s[s_idx] + (1 - lamb) * h_q
            y_mix = lamb * ys[s_idx] + (1 - lamb) * yq

            y_hat_mix = x_mix
            for i in range(2 * l, len(theta), 2):
                y_hat_mix = F.linear(y_hat_mix, theta[i], bias=theta[i + 1])
                y_hat_mix = F.relu(y_hat_mix) if i != (len(theta) - 2) else y_hat_mix

            return y_hat_mix, y_mix
        elif self.model.feature_extractor == 'conv':
            lamb = lamb.unsqueeze(-1).unsqueeze(-1)
            max_l = 4
            l = np.random.randint(1, max_l)

            for i in range(0, 4 * l, 4):
                conv_s = F.conv2d(h_s, weight=theta[i], bias=theta[i+1], padding=1)
                h_s = F.max_pool2d(F.relu(F.batch_norm(conv_s, running_mean=None, running_var=None, weight=theta[i+2], bias=theta[i+3], training=True, momentum=1.0)), 2)

                conv_q = F.conv2d(h_q, weight=theta[i], bias=theta[i+1], padding=1)
                h_q = F.max_pool2d(F.relu(F.batch_norm(conv_q, running_mean=None, running_var=None, weight=theta[i+2], bias=theta[i+3], training=True, momentum=1.0)), 2)

            x_mix = lamb * h_s[s_idx] + (1 - lamb) * h_q
            y_mix = lamb.squeeze(-1).squeeze(-1).squeeze(-1) * ys[s_idx] + (1 - lamb.squeeze(-1).squeeze(-1).squeeze(-1)) * yq

            y_hat_mix = x_mix
            for i in range(4 * l, 4*4, 4):
                y_hat_mix = F.conv2d(y_hat_mix, weight=theta[i], bias=theta[i+1], padding=1)
                y_hat_mix = F.max_pool2d(F.relu(F.batch_norm(y_hat_mix, running_mean=None, running_var=None, weight=theta[i+2], bias=theta[i+3], training=True, momentum=1.0)), 2)

            y_hat_mix = torch.flatten(y_hat_mix, start_dim=1)
            y_hat_mix = F.linear(y_hat_mix, weight=theta[-4], bias=theta[-3])
            y_hat_mix = F.linear(y_hat_mix, weight=theta[-2], bias=theta[-1])

            return y_hat_mix, y_mix


class LAVA(nn.Module):

    def __init__(self, model, steps):
        super().__init__()
        self.model = model
        self.steps = steps

        if model.adaptation == 'full':
            raise NotImplementedError("Lava not implemented for full adaptation")
        else:
            self.theta0 = nn.Parameter(model.initialization)

        self.context_dim = self.model.num_params

        self.reg = 0.1
        self.learning_rate = nn.Parameter(torch.Tensor([0.1]))

        self.last_layer_adapt = isinstance(self.model.main_net.net, LastLayerNet)


    def adapt(self, x, y, steps=None):
        if steps is None:
            steps = self.steps

        def compute_loss_stateless_model(theta0, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)
            loss = self.model.loss_fn(batch, targets, theta0)
            return loss

        ft_compute_grad = torch.func.grad(compute_loss_stateless_model)
        ft_compute_sample_grad = torch.vmap(ft_compute_grad, in_dims=(None, 0, 0))

        z = self.theta0


        # One gradient step to get the MAP
        theta_primes = z

        # Multiple gradient steps
        #for i in range(steps):
        per_sample_grad = ft_compute_sample_grad(theta_primes, x, y).squeeze(1)
        theta_primes = (theta_primes - self.learning_rate * per_sample_grad) #.unsqueeze(1)

        ft_compute_grads_per_theta = torch.vmap(ft_compute_grad, in_dims=(0, 0, 0))
        for i in range(steps - 1):
            per_theta_grad = ft_compute_grads_per_theta(theta_primes, x, y)
            theta_primes = theta_primes - self.learning_rate * per_theta_grad


        if self.last_layer_adapt:
            H1 = self.hessian_last_layer(x, y, theta_primes)
            #H2 = torch.vmap(torch.func.hessian(lambda theta, x, y : compute_loss_stateless_model(theta, x, y)), in_dims=(0, 0, 0))(theta_primes, x, y)
        else:
            H1 = torch.vmap(torch.func.hessian(lambda theta, x, y : compute_loss_stateless_model(theta, x, y)), in_dims=(0, 0, 0))(theta_primes, x, y)

        H = H1
        H = (1 / (1 + self.reg)) * (H + self.reg * torch.eye(self.context_dim).to(x.device))

        H_sum_inv = torch.linalg.inv(torch.sum(H, 0, keepdim=True))
        theta_grad_proj = torch.bmm(H, theta_primes.squeeze(1).unsqueeze(-1))
        final_z = torch.bmm(H_sum_inv, torch.sum(theta_grad_proj, 0, keepdim=True))[:, :, 0].squeeze(0) # (1 x 2)

        return final_z

    def hessian_last_layer(self, x, y, theta_primes):

        batch_size = y.shape[0]

        f = self.model.main_net.net.base_network(x).unsqueeze(1)
        identity = torch.eye(y.shape[-1]).to(x.device)
        H_w = torch.kron(identity, torch.bmm(f.squeeze(1).unsqueeze(-1), f))
        H_b = torch.kron(identity, f)
        h1 = torch.cat([H_w, H_b], 1)
        h2 = torch.cat([H_b.transpose(2, 1), identity.unsqueeze(0).repeat(batch_size, 1, 1)], 1)
        return 2 * torch.cat([h1, h2], -1) / y.shape[-1]


    def forward(self, *args):
        return self.model(*args)

    def criterion(self, ypred, y):
        return self.model.criterion(ypred, y)

    def optimize(self, opt, loss, xq=None, yq=None, reg_loss=0):
        self.model.optimize(opt, loss, reg_loss=reg_loss)


        