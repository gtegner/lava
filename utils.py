import os

def generate_model_name(args):
    return f"{args.model_name}-model-{args.model}-{args.adaptation}-context-{args.context_dim}-steps-{args.steps}-dataset-{args.dataset}-num-tasks-{args.num_params}-supp-sz-{args.support_size}-noise-{args.noise_std}-traj-{args.use_trajectory}-seed-{args.seed}"

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)