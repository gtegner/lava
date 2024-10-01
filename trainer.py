import torch
import pickle
import os
import time
from collections import defaultdict
import numpy as np


class Trainer(object):
    def __init__(self, args, save_folder, model_path, use_trajectory, device, writer=None):
        super().__init__()
        self.save_folder = save_folder
        self.model_path = model_path
        self.use_trajectory = use_trajectory
        
        self.device = device

        self.results = dict()
        self.writer = writer
        self.args = args

        self.test_freq = 5
        self.best_model = None

        if args.dataset == 'imagenet':
            self.mode = 'classification'
            self.test_freq = 1
        else:
            self.mode = 'regression'


    def fit(self, epochs, model, dloader, opt, dloader_test=None):
        results_train = defaultdict(dict)
        results_test = defaultdict(dict)

        best_loss = np.inf

        for epoch in range(1, epochs + 1):
            self.train(epoch, model, dloader, opt, 'train', results_train)

            if epoch % self.test_freq == 0 or epoch == 1:
                test_loss = self.train(epoch, model, dloader_test, opt, 'test', results_test)
                if test_loss < best_loss:
                    best_model = model
                    best_loss = test_loss
                    self.best_model = best_model

                # try:
                #     if hasattr(dloader.dataset, 'plot_predictions_trajectory'):
                #         dloader.dataset.plot_predictions_trajectory(model, os.path.join(self.save_folder, 'figures', f'preds_{epoch}.png'), self.device)
                # except Exception as e:
                #     print("Can't plot predictions")


            self.save_results(os.path.join(self.save_folder, 'results_train.pkl'), results_train)
            self.save_results(os.path.join(self.save_folder, 'results_test.pkl'), results_test)

    def save_results(self, path, dict):
        with open(path, 'wb') as f:
            pickle.dump(dict, f)

    def eval(self, model, dset, all_eval_results):

        results_dict = defaultdict(dict)
        dloader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=True)
        self.train(0, model, dloader, None, f'test_{dset.id}', results_dict)
        all_eval_results[dset.id] = results_dict

        self.save_results(os.path.join(self.save_folder,'eval_results.pkl'), all_eval_results)
        
    def get_metrics(self, ypred, y):
        if self.mode == 'classification':
            ypred = ypred[-1] if isinstance(ypred, list) else ypred
            classes = torch.argmax(ypred, -1)
            num_correct = torch.sum((classes == y) * 1)
            acc = num_correct / y.shape[0]
            return acc.cpu().data.numpy()
        elif self.mode == 'regression':
            return 0


    def train(self, epoch, model, dloader, opt, phase, results_dict=None):

        mu_loss = 0
        mu_time = 0
        mu_accuracy = 0
        for ix, (xs_batch, ys_batch, xq_batch, yq_batch) in enumerate(dloader):


            start = time.time()
            num_tasks = xs_batch.shape[0]

            loss = [] #0
            reg_loss = 0
            accuracy = 0

            #model.zero_grad()
            for i in range(num_tasks):
                xs = xs_batch[i].to(self.device)
                ys = ys_batch[i].to(self.device)
                xq = xq_batch[i].to(self.device)
                yq = yq_batch[i].to(self.device)

                #model.zero_grad()
                z = model.adapt(xs, ys)

                # if self.use_trajectory:
                #     ypred = model(yq[0], xq, z)
                # else:
                ypred = model(xq, z)

                if hasattr(model, 'metamix') and phase == 'train':
                    ypred, yq = model.mix(xs, ys, xq, yq, z)

                query_loss = model.criterion(ypred, yq)

                if hasattr(model, 'det_reg') and phase == 'train':
                    if model.det_reg:
                        reg_loss += model.get_H_reg(xq, yq, z)

                #loss += query_loss
                loss.append(query_loss)

                accuracy += self.get_metrics(ypred, yq)

            #loss /= num_tasks
            
            loss = [torch.cat([x[0].view(1) for x in loss], -1), torch.cat([x[1].view(1) for x in loss], -1)] if isinstance(loss[0], list) else loss
            loss = [x.mean() for x in loss] if loss[0].dim() > 0 else sum(loss) / num_tasks

            if phase == 'train':
                model.optimize(opt, loss, xq=xq_batch, yq=yq_batch, reg_loss=reg_loss)

            if isinstance(loss, list):
                loss = torch.cat([torch.unsqueeze(x, 0) for x in loss])
                loss = torch.mean(loss, -1)

            loss_item = loss.item() if loss.dim() == 0 else loss[-1].item()

            mu_loss += loss_item

            accuracy /= num_tasks
            mu_accuracy += accuracy

            end = time.time()
            total_time = (end - start) / num_tasks

            mu_time += total_time

            if self.mode == 'classification':
                print(f"Epoch: {epoch} Loss: {loss_item:.3f}")


        mu_loss /= len(dloader)
        mu_time /= len(dloader)
        mu_accuracy /= len(dloader)

        if results_dict is not None:
            results_dict['compute_time'] = total_time
            results_dict['loss'][epoch] = mu_loss
            results_dict['accuracy'][epoch] = mu_accuracy

        if self.writer is not None:
            self.writer.add_scalar(f'{phase}/loss', mu_loss, epoch)
            self.writer.add_scalar(f'{phase}/accuracy', mu_accuracy, epoch)
            self.writer.add_scalar(f'{phase}/compute_time', mu_time, epoch)

        if epoch % 10 == 0:
            print(f"{phase.upper()} Epoch: {epoch} Loss: {loss_item:3f}")  #Compute: {total_time:1e}")
            torch.save(model, self.model_path)

        if epoch == 1:
            print(f"Compute: {mu_time}")
        return mu_loss

