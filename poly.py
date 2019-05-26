from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Optuna settings
STUDY_NAME = 'poly'
N_TRIALS = 10
PRUNER_INTERVAL = 50

# Machine learning settings
EPOCH = 1000
DATA_SIZE = 1000
BATCH_SIZE = 100
GPU_ID = -1  # Set value >= 0 to use GPU (-1: CPU mode)

# Others
DATASET_DIRECTORY = Path(f"./data/dataset_{DATA_SIZE}")
MODEL_DIRECTORY = Path('models/poly')


def generate_data(size=1000):
    """Generate training data.

    Args:
        length: int
            The sample size of the data.
    Returns:
        x_train: numpy.ndarray
            The input data for training.
        y_train: numpy.ndarray
            The output data for training.
        x_valid: numpy.ndarray
            The input data for validation.
        y_valid: numpy.ndarray
            The output data for validation.
    """
    x = np.random.rand(size, 1).astype(np.float32) * 2 * np.pi
    y = np.sin(x)
    n_train = int(size * 0.8)

    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def prepare_dataset():
    """Prepare dataset for optimization."""
    if not DATASET_DIRECTORY.exists():
        DATASET_DIRECTORY.mkdir(parents=True)
        x_train, y_train, x_valid, y_valid = generate_data(DATA_SIZE)
        np.save(DATASET_DIRECTORY / 'x_train.npy', x_train)
        np.save(DATASET_DIRECTORY / 'y_train.npy', y_train)
        np.save(DATASET_DIRECTORY / 'x_valid.npy', x_valid)
        np.save(DATASET_DIRECTORY / 'y_valid.npy', y_valid)


def objective(trial):
    """Objective function to make optimization for Optuna.

    Args:
        trial: optuna.trial.Trial
    Returns:
        None
    """

    # Suggest hyperparameters
    polynomial_degree = trial.suggest_int('polynomial_degree', 1, 10)

    print('--')
    print(f"Trial: {trial.number}")
    print('Current hyperparameters:')
    print(f"    Polynomial degree: {polynomial_degree}")
    print('--')

    # Generate the model
    model = make_pipeline(PolynomialFeatures(polynomial_degree), Ridge())

    # Create dataset
    x_train = np.load(DATASET_DIRECTORY / 'x_train.npy')
    y_train = np.load(DATASET_DIRECTORY / 'y_train.npy')
    x_valid = np.load(DATASET_DIRECTORY / 'x_valid.npy')
    y_valid = np.load(DATASET_DIRECTORY / 'y_valid.npy')

    # Train
    model.fit(x_train, y_train)

    # Save model
    with open(MODEL_DIRECTORY / f"model_{trial.number}.pickle", 'wb') as f:
        pickle.dump(model, f)

    # Evaluate loss
    loss = np.mean((model.predict(x_valid) - y_valid)**2)**.5
    return loss


def evaluate_results(trial):
    """Evaluate the optimization results.

    Args:
        study: optuna.trial.Trial
    Returns:
        None
    """

    # Load model
    trial_number = trial.number
    with open(
            MODEL_DIRECTORY / f"model_{trial_number}.pickle", 'rb') as f:
        model = pickle.load(f)

    # Load data
    x_valid = np.load(DATASET_DIRECTORY / 'x_valid.npy')
    y_valid = np.load(DATASET_DIRECTORY / 'y_valid.npy')

    # Plot
    plt.plot(x_valid, y_valid, '.', label='answer')
    plt.plot(x_valid, model.predict(x_valid), '.', label='prediction')
    plt.legend()
    plt.show()


def main():
    # Generate dataset
    prepare_dataset()
    if not MODEL_DIRECTORY.exists():
        MODEL_DIRECTORY.mkdir(parents=True)

    # Prepare study
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_NAME}.db",
        load_if_exists=True, pruner=optuna.pruners.MedianPruner())

    # Optimize
    study.optimize(objective, n_trials=N_TRIALS)

    # Visualize the best result
    print('=== Best Trial ===')
    print(study.best_trial)
    evaluate_results(study.best_trial)


if __name__ == '__main__':
    main()
