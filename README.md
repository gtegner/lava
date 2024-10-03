
# 🌋 Reducing Variance in Meta-Learning via Laplace Approximation for Regression Tasks (LAVA)
This repository contains a minimal implementation of "Reducing Variance in Meta-Learning via Laplace Approximation for Regression Tasks". 

***Authors***: Alfredo Reichlin*, Gustaf Tegnér*, Miguel Vasco, Hang Yin, Mårten Björkman, and Danica Kragic

<center>
<img src="lava.png" alt="Description of image" width="600" height="auto"/>
</center>


## Installation

The codebase is implemented in Python 3.9. You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Experiments
To train a model, variations of the following command is used:

```python main.py --model-name {PREFIX} --dataset {DATASET} --model {MODEL_NAME} --steps {STEPS} --adaptation {ADAPTATION} --context-dim {CONTEXT_DIM} --support-size {SUPPORT_SIZE}```

An explanation of the parameters follows below:
```
model-name: prefix of the directory to save the model. By default, models are saved in the `checkpoints/{model_name}` directory.

dataset: [mass-spring, fitz, vanderpol, pendulum, sine]

model: ['lava', 'maml', 'vr-maml', 'metamix', 'vfml', 'llama']

steps: Number of steps to perform in the inner-loop update

adaptation: One of ['full', 'conditional', 'head'] which considers updating all parameters (MAML), using conditioning (CAVIA) or only updated the last layer (ANIL). Currently LAVA does not suppport the adaptation over the full parameters.

context-dim: Dimensions when using conditioning (adaptation == 'conditional'), ignored otherwise

support-size: Number of samples in the support-data
```

An example on how to train LAVA on the sine data is:

```
python main.py --model-name readme --dataset sine --model lava --steps 1 --adaptation head --support-size 10
```

## Evaluation
After running experiments, evaluation results are saved in the `checkpoints/{model_name}` directory. Each model's directory contains `results_train.pkl` and `results_test.pkl` which contain the training and testing results respectively. The metrics are also logged in the tensorboard directory `tensorboard/{dataset}` by default.


## Citation
If you find this repository useful, please consider citing our work:
```
@misc{reichlin2024reducingvariancemetalearninglaplace,
      title={Reducing Variance in Meta-Learning via Laplace Approximation for Regression Tasks}, 
      author={Alfredo Reichlin and Gustaf Tegnér and Miguel Vasco and Hang Yin and Mårten Björkman and Danica Kragic},
      year={2024},
      eprint={2410.01476},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01476}, 
}
```
