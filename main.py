import numpy as np
import torch

import argparse
import torch.nn.functional as F
from trainer import Trainer
import os
import utils
from datasets.meta_dataset import MetaODEDataset
from datasets.ode_bank import Sine, FitzHughNagumo, VanDerPolOscillator, PendulumODE, MassSpringODE
from models.gbml_baselines import MAMLOptimizer, LAVA, VR_MAML, VFML, METAMIX
from models.ode_wrapper import ODEWrapper
from collections import defaultdict
from models.base_models import VectorField


import torch
from torch.utils.tensorboard import SummaryWriter


'''
    runnable models:
    'lava', 'maml', 'llama', 'vr-maml', 'vfml', 'metamix'
'''

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default='exp')
parser.add_argument('--model', type=str, default='metamix')
parser.add_argument('--dataset', type=str, default='sine')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
parser.add_argument('--tensorboard-dir', type=str, default='tensorboard')

parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--support-size', type=int, default=10)
parser.add_argument('--noise-std', type=float, default=0)

parser.add_argument('--query-size', type=int, default=50)
parser.add_argument('--num-params', type=int, default=100)

parser.add_argument('--context-dim', type=int, default=16)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--adaptation', type=str, default='full')
parser.add_argument('--use-trajectory', type=int, default=0)

args = parser.parse_args()


# Set up model directories
if args.model == 'vr-maml' or args.model == 'metamix':
    args.adaptation = "full"

model_name = utils.generate_model_name(args)

writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.dataset, model_name))

MODEL_DIR = os.path.join(args.checkpoints_dir, model_name)
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')
FIGURES_DIR= os.path.join(MODEL_DIR, 'figures')

args.model_dir = MODEL_DIR

utils.create_dir(args.checkpoints_dir)
utils.create_dir(MODEL_DIR)
utils.create_dir(FIGURES_DIR)


# Save parameters
import pickle
with open(os.path.join(MODEL_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump({'args': args}, f)


# Set up dataset
ODE_DATASETS = ['fitz', 'vanderpol', 'pendulum', 'mass-spring', 'sine']

if args.dataset in ODE_DATASETS:
    feature_extractor = 'nn'
    objective = 'regression'
    if args.dataset == 'fitz':
        ode = FitzHughNagumo(0.1, 10)
    elif args.dataset == 'vanderpol':
        ode = VanDerPolOscillator(0.1, 10)
    elif args.dataset == 'pendulum':
        ode = PendulumODE(0.1, 10)
    elif args.dataset == 'mass-spring':
        ode = MassSpringODE(0.1, 10)
    elif args.dataset == 'sine':
        ode = Sine(0.1, 10)

    dset = MetaODEDataset(ode, args.num_params, args.support_size, args.query_size, args.use_trajectory, phase='train', noise_std=args.noise_std, seed=args.seed)
    dset_test = MetaODEDataset(ode, args.num_params, args.support_size, args.query_size, args.use_trajectory, phase='test', std=dset.std, noise_std=args.noise_std, seed=args.seed+1)


dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
dloader_test = torch.utils.data.DataLoader(dset_test, batch_size=32, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
xs, ys, xq, yq = next(iter(dloader))
input_dim = dset.input_dim
output_dim = dset.output_dim

model = VectorField(input_dim, output_dim, adaptation=args.adaptation, context_dim=args.context_dim, feature_extractor=feature_extractor, objective=objective)

if args.use_trajectory and args.dataset not in ['imagenet']:
    model = ODEWrapper(model)


if args.model == 'lava':
    model = LAVA(model, steps=args.steps)
elif args.model in ['maml', 'llama']:
    det_reg = True if args.model == 'llama' else False
    model = MAMLOptimizer(model, args.steps, det_reg=det_reg)
elif args.model == 'vr-maml':
    model = VR_MAML(model, args.steps)
elif args.model == 'vfml':
    model = VFML(model, args.steps)
elif args.model == 'metamix':
    model = METAMIX(model, args.steps)
else:
    raise NotImplementedError(f"{args.model} not implemented")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model = model.to(device)
# Fit the model
trainer = Trainer(args=args, save_folder=MODEL_DIR, model_path=MODEL_PATH, use_trajectory=args.use_trajectory, device=device, writer=writer)
trainer.fit(args.epochs, model, dloader, opt, dloader_test=dloader_test)
