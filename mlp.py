import glob
import os
from pathlib import Path

import chainer as ch
import matplotlib.pyplot as plt
import numpy as np
import optuna


# Optuna settings
STUDY_NAME = 'mlp'
N_TRIALS = 100
PRUNER_INTERVAL = 100

# Machine learning settings
EPOCH = 5000
DATA_SIZE = 1000
BATCH_SIZE = 100
GPU_ID = -1  # Set value >= 0 to use GPU (-1: CPU mode)

# Others
DATASET_DIRECTORY = Path(f"./data/dataset_{DATA_SIZE}")
MODEL_DIRECTORY = Path('models/mlp')


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


def generate_model(trial):
    """Generate MLP model.

    Args:
        trial: optuna.trial.Trial
    Returns:
        classifier: chainer.links.Classifier
    """
    # Suggest hyperparameters
    layer_number = trial.suggest_int('layer_number', 2, 5)
    activation_name = trial.suggest_categorical(
        'activation_name', ['relu', 'sigmoid'])
    unit_numbers = [
        trial.suggest_int(f"unit_number_layer{i}", 10, 100)
        for i in range(layer_number - 1)] + [1]
    dropout_ratio = trial.suggest_uniform('dropout_ratio', 0.0, 0.2)

    print('--')
    print(f"Trial: {trial.number}")
    print('Current hyperparameters:')
    print(f"    The number of layers: {layer_number}")
    print(f"    Activation function: {activation_name}")
    print(f"    The number of units for each layer: {unit_numbers}")
    print(f"    The ratio for dropout: {dropout_ratio}")
    print('--')

    # Generate the model
    model = MLP(
        unit_numbers, activation_name=activation_name,
        dropout_ratio=dropout_ratio)
    classifier = ch.links.Classifier(
        model, lossfun=ch.functions.mean_squared_error)
    classifier.compute_accuracy = False
    return classifier


def objective(trial):
    """Objective function to make optimization for Optuna.

    Args:
        trial: optuna.trial.Trial
    Returns:
        loss: float
            Loss value for the trial
    """

    # Generate model
    classifier = generate_model(trial)

    # Create dataset
    x_train = np.load(DATASET_DIRECTORY / 'x_train.npy')
    y_train = np.load(DATASET_DIRECTORY / 'y_train.npy')
    x_valid = np.load(DATASET_DIRECTORY / 'x_valid.npy')
    y_valid = np.load(DATASET_DIRECTORY / 'y_valid.npy')

    # Prepare training
    train_iter = ch.iterators.SerialIterator(
        ch.datasets.TupleDataset(x_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True)
    valid_iter = ch.iterators.SerialIterator(
        ch.datasets.TupleDataset(x_valid, y_valid),
        batch_size=BATCH_SIZE, shuffle=False, repeat=False)

    optimizer = ch.optimizers.Adam()
    optimizer.setup(classifier)
    updater = ch.training.StandardUpdater(train_iter, optimizer, device=GPU_ID)

    stop_trigger = ch.training.triggers.EarlyStoppingTrigger(
        monitor='validation/main/loss', check_trigger=(100, 'epoch'),
        max_trigger=(EPOCH, 'epoch'))

    trainer = ch.training.Trainer(
        updater, stop_trigger,
        out=MODEL_DIRECTORY/f"model_{trial.number}")
    log_report_extension = ch.training.extensions.LogReport(
        trigger=(100, 'epoch'), log_name=None)
    trainer.extend(log_report_extension)
    trainer.extend(ch.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(ch.training.extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'))
    trainer.extend(ch.training.extensions.Evaluator(valid_iter, classifier))
    trainer.extend(ch.training.extensions.ProgressBar())
    trainer.extend(
        optuna.integration.ChainerPruningExtension(
            trial, 'validation/main/loss', (PRUNER_INTERVAL, 'epoch')))

    # Train
    trainer.run()

    loss = log_report_extension.log[-1]['validation/main/loss']
    return loss


class MLP(ch.ChainList):
    """Multi Layer Perceptron."""

    def __init__(
            self, unit_numbers, activation_name, dropout_ratio):
        """Initialize MLP object.

        Args:
            unit_numbers: list of int
                List of the number of units for each layer.
            activation_name: str
                The name of the activation function applied to layers except
                for the last one (The activation of the last layer is always
                identity).
            dropout_ratio: float
                The ratio of dropout. Dropout is applied to all layers.
        Returns:
            None
        """
        super().__init__(*[
            ch.links.Linear(unit_number)
            for unit_number in unit_numbers])
        self.activations = [
            self._create_activation_function(activation_name)
            for _ in self[:-1]] \
            + [ch.functions.identity]  # The last one is identity
        self.dropout_ratio = dropout_ratio

    def _create_activation_function(self, activation_name):
        """Create activation function.

        Args:
            activation_name: str
                The name of the activation function.
        Returns:
            activation_function: chainer.FunctionNode
                Chainer FunctionNode object corresponding to the input name.
        """
        if activation_name == 'relu':
            return ch.functions.relu
        elif activation_name == 'sigmoid':
            return ch.functions.sigmoid
        elif activation_name == 'identity':
            return ch.functions.identity
        else:
            raise ValueError(f"Unknown function name {activation_name}")

    def __call__(self, x):
        """Execute the NN's forward computation.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
        Returns:
            y: numpy.ndarray or cupy.ndarray
                Output of the NN.
        """
        h = x
        for i, link in enumerate(self):
            h = link(h)
            if i + 1 != len(self):
                h = ch.functions.dropout(h, ratio=self.dropout_ratio)
            h = self.activations[i](h)
        return h


def evaluate_results(trial):
    """Evaluate the optimization results.

    Args:
        study: optuna.trial.Trial
    Returns:
        None
    """
    # Load model
    trial_number = trial.number
    unit_numbers = []
    for i in range(100):
        param_key = f"unit_number_layer{i}"
        if param_key not in trial.params:
            break
        unit_numbers.append(trial.params[param_key])
    model = MLP(
        unit_numbers + [1], trial.params['activation_name'],
        trial.params['dropout_ratio'])
    snapshots = glob.glob(str(MODEL_DIRECTORY / f"model_{trial_number}" / '*'))
    snapshot = max(snapshots, key=os.path.getctime)
    print(f"Loading: {snapshot}")
    ch.serializers.load_npz(
        snapshot, model, path='updater/model:main/predictor/')

    # Load data
    x_valid = np.load(DATASET_DIRECTORY / 'x_valid.npy')
    y_valid = np.load(DATASET_DIRECTORY / 'y_valid.npy')

    # Plot
    plt.plot(x_valid, y_valid, '.', label='answer')
    with ch.using_config('train', False):
        predict = model(x_valid).data
    plt.plot(x_valid, predict, '.', label='prediction')
    plt.legend()
    plt.show()


def main():
    # Generate dataset
    prepare_dataset()

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
