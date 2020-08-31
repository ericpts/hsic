#!/usr/bin/env python3
import lib_biased_mnist
from pathlib import Path
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)


from lib_experiment import Experiment, grid_param, fixed_param

LABEL_CORRELATIONS = [0.999, 0.99, 0.9, 0.8]
NOISE_LEVELS = [0]

experiment = Experiment(
    name=Path(__file__).stem,
    problem="biased_mnist",
    n_runs=3,
    f_pre_experiment=lambda: lib_biased_mnist.regenerate_all_data(
        LABEL_CORRELATIONS, NOISE_LEVELS
    ),
    experiment_dict={
        "diversity_loss.independence_measure": fixed_param("cka"),
        "diversity_loss.kernel": fixed_param("rbf"),
        "compute_combined_loss.diversity_loss_coefficient": grid_param(
            [0, 1, 2, 4, 8, 16, 64, 256,]
        ),
        "BiasedMnistProblem.training_data_label_correlation": grid_param(
            LABEL_CORRELATIONS
        ),
        "BiasedMnistProblem.filter_for_digits": fixed_param(list(range(10))),
        "BiasedMnistProblem.model_type": fixed_param("mlp"),
        "BiasedMnistProblem.background_noise_level": grid_param(NOISE_LEVELS),
        "Problem.n_epochs": fixed_param(100),
        "Problem.initial_lr": grid_param([0.001, 0.01]),
        "Problem.decrease_lr_at_epochs": fixed_param([20, 40, 80]),
        "Problem.n_models": fixed_param(2),
        "Problem.optimizer": fixed_param("@tf.keras.optimizers.Nadam"),
        "tf.keras.optimizers.Nadam.epsilon": fixed_param(0.001),
        "get_weight_regularizer.strength": fixed_param(0.01),
    },
)

