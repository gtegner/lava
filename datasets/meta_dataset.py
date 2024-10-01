import torch
import numpy as np
from datasets.ode_utils import _numerical_diff, to_torch
import matplotlib.pyplot as plt
import os
from torchdiffeq import odeint

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().data.numpy()
    return tensor


def _support_query_split(x, y, support_size, query_size, sequential=False, mix=False):
    indices = np.arange(len(x))
    if not sequential:
        np.random.shuffle(indices)

    support_indices = indices[0 : support_size]
    query_indices = indices[support_size : support_size + query_size]

    if mix:
        indices2 = np.arange(len(x))
        np.random.shuffle(indices2)
        support_indices = indices2[0 : support_size]


    xs = x[support_indices]
    ys = y[support_indices]
    xq = x[query_indices]
    yq = y[query_indices]

    return xs, ys, xq, yq


class MetaODEDataset(torch.utils.data.Dataset):
    def __init__(self, ode, num_params, support_size, query_size, generate_traj, phase, std=None, noise_std=0, seed=0, **kwargs):
        super().__init__()

        self.id = f"{ode.name}-num-params={num_params}-support-size={support_size}-noise-std={noise_std}-seed={seed}"

        self.noise_std = noise_std

        self.support_size = support_size
        self.query_size = query_size
        total_size = self.support_size + self.query_size

        self.rng = np.random.default_rng(seed)

        self.generate_traj = generate_traj
        self.ode = ode

        params = self.generate_params(num_params)

        #initial = self.generate_x(num_params, total_size) # num_points per param
        data = []
        
        self.input_dim = self.ode.dim
        self.output_dim = self.ode.dim

        for ix, param in enumerate(params):

            #x0 = initial[ix]
            x0 = self.generate_x(total_size, param=param)

            t_range = self.ode.get_t_range(param)

            data_point = self.generate_data_point(x0, t_range, param)
            data.append(data_point)

        self.data = data

        if phase == 'train':
            self.std = self.get_std(self.data)
        elif phase == 'test' and std is not None:
            self.std = std
    
    def get_std(self, data):
        all_targets = []
        for point in data:
            if self.generate_traj:
                ys = point['trajectory']
            else:
                ys = point['vector_field_from_initial']
            all_targets.append(to_numpy(ys))

        all_targets = np.array(all_targets)
        all_targets = all_targets.reshape(-1, self.output_dim)
        std = np.std(all_targets, 0)
        return std


    def __getitem__(self, idx, with_params=False, sequential=False, mix=False):

        point = self.data[idx]
        
        param = point['param']

        if self.generate_traj:
            # Support query should be different trajectories here

            xy = point['trajectory']
            times = point['times']
            vector_field = point['vector_field']

            ys = xy[:, np.random.randint(xy.shape[1]), :] # Random traj
            yq = xy[:, np.random.randint(xy.shape[1]), :] # Random traj
            xs = times
            xq = times
            #xs, ys, xq, yq = _support_query_split(times, xy, self.support_size, self.query_size, sequential=True)

        else:
            xy = point['initial']
            vector_field = point['vector_field_from_initial']
            xs, ys, xq, yq = _support_query_split(xy, vector_field, self.support_size, self.query_size, sequential=sequential, mix=mix)

        # Noise on support only
        ys = ys + torch.randn_like(ys) * self.noise_std * self.noise_std

        if with_params:
            return xs, ys, xq, yq, param

        return xs, ys, xq, yq #, param

    def __len__(self):
        return len(self.data)

    def generate_data_point(self, initial, t_range, param):

        vector_field_on_initial = self.ode.vector_field(initial, np.zeros(len(initial)), param)


        
        trajectory = None
        estimated_vector_field = None
        gt_vector_field = None

        if self.generate_traj:
            trajectory = self.ode.solve(initial, t_range, param)
            estimated_vector_field = _numerical_diff(trajectory, t_range)
            gt_vector_field = self.ode.vector_field(trajectory, t_range, param)

        data_point = {
            'initial': initial,
            'vector_field_from_initial': vector_field_on_initial,

            'times': t_range,
            'trajectory': trajectory,
            'vector_field': gt_vector_field,
            'estimated_vector_field': estimated_vector_field,

            'param': param
        }

        return data_point

    def vector_field(self, x, t):
        return self.ode(x, t)

    def generate_params(self, num_params):
        return self.ode.generate_params(num_params)

    def generate_x(self, *size, param=None):
        points =  self.ode.sample_from_domain(*size, param=param)
        return to_torch(points)

    def plot_predictions_vector_field(self, model, save_path, device):
        #xs, ys, xq, yq = self.__getitem__(np.random.randint(self.__len__()))
        xs, ys, xq, yq = self.__getitem__(int(self.__len__() // 10))


        xs, ys, xq, yq = xs.to(device), ys.to(device), xq.to(device), yq.to(device)
        z = model.adapt(xs, ys)

        pred_traj = model(xq, z).detach().cpu()
        xq = xq.detach().cpu()
        yq = yq.detach().cpu()

        def plot_traj(x, pred, gt):
            fig, ax = plt.subplots()
            ax.quiver(x[:,0], x[:,1], pred[:,0], pred[:,1], label='predictions', color='tab:orange', alpha=0.5)
            ax.quiver(x[:,0], x[:,1], gt[:,0], gt[:,1], label='gt', color='tab:blue')
            ax.legend()
            return fig

        if pred_traj.shape[-1] == 2 and xs.shape[-1] == 2:
            fig = plot_traj(xq, pred_traj, yq)
            fig.savefig(save_path)
            plt.close(fig)
            plt.close('all')


    def plot_trajectory(self, model, save_path, device, xs, ys, xq, yq, params, ax=None):
        xs, ys, xq, yq = xs.to(device), ys.to(device), xq.to(device), yq.to(device)
        z = model.adapt(xs, ys)


        t_range = torch.linspace(0, 2, 20)

        y_pred = odeint(lambda t, x : model(x, z), xq[0], t=t_range)
        #y_true = torch.odeint(lambda t, x : self.ode.vector_field(x, params), xq[0], t=t_range)
        y_true = self.ode.solve(xq[0], t_range, params)

        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()

        def plot_traj(pred, gt, ax=None):
            ax.scatter(pred[:,0], pred[:,1], label='predictions', color='tab:orange', alpha=0.5)
            ax.scatter(gt[:,0], gt[:,1], label='gt', color='tab:blue')
            #ax.legend()


        if xs.shape[-1] == 2 and ys.shape[-1] == 2:
            if ax is None:
                fig, ax = plt.subplots()

            plot_traj(y_pred, y_true, ax=ax)

            if ax is None:
                fig.savefig(save_path)
                plt.close(fig)
                plt.close('all')

    def get_predictions_trajectory(self, model, device, index):

        xs, ys, xq, yq, params = self.__getitem__(index, with_params=True, sequential=True, mix=True)
        xs, ys, xq, yq = xs.to(device), ys.to(device), xq.to(device), yq.to(device)
        z = model.adapt(xs, ys)

        preds = []
        trues = []
        initials = []
        t_range = torch.linspace(0, 2, 20)
        if self.ode.name in ['vanderpol']:
            t_range = torch.linspace(0,2, 40)

        for i in range(10):
            y_pred = odeint(lambda t, x : model(x, z), xq[i], t=t_range).cpu().detach()
            preds.append(y_pred)
            y_true = self.ode.solve(xq[i], t_range, params).cpu().detach()
            trues.append(y_true)

            xq_new = xq.clone().cpu().detach()
            initials.append(xq_new[i])

        preds = torch.stack(preds)
        trues = torch.stack(trues)
        initials = torch.stack(initials)

        return initials, trues, preds

    def plot_predictions_trajectory(self, model, save_path, device):
        fig, ax = plt.subplots()
        i = 0
        xs, ys, xq, yq, params = self.__getitem__(i, with_params=True)
        xs, ys, xq, yq = xs.to(device), ys.to(device), xq.to(device), yq.to(device)

        for i in range(10):
            z = model.adapt(xs, ys)
            t_range = torch.linspace(0, 2, 20)
            y_pred = odeint(lambda t, x : model(x, z), xq[i], t=t_range).cpu().detach()
            y_true = self.ode.solve(xq[i], t_range, params).cpu().detach()

            xq_new = xq.clone().cpu().detach()

            def plot_traj(xq, pred, gt, ax=None):
                ax.scatter(pred[:,0], pred[:,1], label='predictions', color='tab:orange', alpha=0.5)
                ax.scatter(gt[:,0], gt[:,1], label='gt', color='tab:blue')
                ax.scatter(xq[i,0], xq[i,1], color='red', s=80)

            if xs.shape[-1] == 2 and ys.shape[-1] == 2:
                plot_traj(xq_new, y_pred, y_true, ax=ax)


        fig.savefig(save_path)
        plt.close(fig)
        plt.close('all')


    def viz(self, save_dir):

        fig, axes = plt.subplots(ncols=4, figsize=(4 * 4, 3))
        
        for i in range(4):
            ax = axes[i]
            xs, ys, xq, yq = self.__getitem__(np.random.randint(self.__len__()))


            if xs.shape[-1] == 1 and ys.shape[-1] == 1:

                ax.scatter(xq[:,0], yq[:,0], label='query', alpha=0.5, color='tab:orange')
                ax.scatter(xs[:,0], ys[:,0], label='support', color='tab:blue')

                ax.set_xlim(-5, 5)

                ax.legend()
                #path = os.path.join(save_dir, 'viz.png')
            elif ys.shape[-1] == 2:
                # This is a vector field
                #ax.scatter(yq[:,0], yq[:,0], label='query', alpha=0.5, color='tab:orange')
                #ax.scatter(ys[:,0], ys[:,0], label='support', color='tab:blue')
                ax.quiver(xs[:,0], xs[:,1], ys[:,0], ys[:,1], label='support')
                ax.quiver(xq[:,0], xq[:,1], yq[:,0], yq[:,1], label='query')


        fig.savefig(os.path.join(save_dir, 'viz.png'))

        plt.close(fig)
        plt.close('all')


