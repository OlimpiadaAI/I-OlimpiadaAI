import pickle
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import argparse


parser = argparse.ArgumentParser(description='Run the script with or without training.')
parser.add_argument('--train', action='store_true', help='Enable training mode.')
args = parser.parse_args()
TRAIN = args.train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")


if TRAIN:
    print("Training the model. This should finish in less than 5 minutes.")
    python_script_filename = "zadanie_jako_skrypt.py"

    # Command to convert .ipynb to .py
    command = [
        "jupyter",
        "nbconvert",
        "--to",
        "script",
        "--output",
        python_script_filename.split(".")[0],
        "pruning.ipynb",
    ]

    subprocess.run(command)

    with open(python_script_filename, "r") as file:
        filedata = file.read()
        filedata = re.sub(
            "FINAL_EVALUATION_MODE = False",
            "FINAL_EVALUATION_MODE = True",
            filedata,
        )

    with open(python_script_filename, "w") as file:
        file.write(filedata)

    # execute the python file
    vars = {
        "script": str(Path(python_script_filename).name),
        "__file__": str(Path(python_script_filename)),
    }
    exec(open(python_script_filename).read(), vars)

################ Validation #############################################


class MLP(nn.Module):
    """Multi Layer Perceptron with one hidden layer."""

    def __init__(self, *args):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

    def loss(self, input, target, reduction="mean"):
        mse_loss = nn.MSELoss(reduction=reduction)
        return mse_loss(input, target)


def load_parameters(model, file_name="model_parameters.pkl", from_file=True, params=None):

    if from_file:
        with open(f"{file_name}", "rb") as f:
            params_to_load = pickle.load(f)
    else:
        params_to_load = params
        
    for name, param in model.named_parameters():
        with torch.no_grad():
            param[...] = params_to_load[name].to(device)


class InMemDataset(Dataset):
    """Load the data into the memory. This class inherits from Dataset."""

    def __init__(self, xs, ys, device="cpu"):
        super().__init__()
        self.dataset = []
        for i in tqdm(range(len(xs))):
            self.dataset.append(
                (
                    torch.tensor(xs[i]).to(device).float(),
                    torch.tensor(ys[i]).to(device).float(),
                )
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def load_test_data(
    x_test_path,
    y_test_path,
):

    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    return X_test, y_test


def compute_error(model, data_loader):
    """Calculate the MSE in the data_loader."""
    model.eval()

    losses = 0
    num_of_el = 0
    with torch.no_grad():
        for x, y in data_loader:
            outputs = model(x)
            num_of_el += x.shape[0] * y.shape[1]
            losses += model.loss(outputs, y, reduction="sum")

    return losses / num_of_el


def score(mse_loss, sparsity, mse_weight=1.5, sparsity_weight=1.5):

    if type(mse_loss) == np.ndarray:
        mse_loss[mse_loss > 1000] = 1000
    else:
        if mse_loss > 1000:
            mse_loss = 1000

    score = (1 - mse_loss / 1000) ** mse_weight * sparsity**sparsity_weight
    return score


def points(score):
    def scale(x, lower=0.085, upper=0.95, max_points=1.5):
        scaled = min(max(x, lower), upper)
        return (scaled - lower) / (upper - lower) * max_points
    return scale(score)


def get_sparsity(model):
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

    sparsity = zero_params / total_params
    return sparsity


def evaluate_algorithm(X_test, y_test):
    batch_size = 128

    test_model = MLP().to(device)
    load_parameters(test_model)

    _test = InMemDataset(X_test, y_test, device)
    test_loader = DataLoader(_test, batch_size=batch_size, shuffle=False)

    mse = compute_error(test_model, test_loader)
    sparsity = get_sparsity(test_model)
    model_score = score(mse, sparsity)

    return model_score


X_test, y_test = load_test_data(
    "valid_data/X_valid.npy", "valid_data/y_valid.npy"
)

model_score = evaluate_algorithm(X_test, y_test)

print(f"This model got score {model_score:.3f} on validation set!")
print(f"This model got {points(model_score):.3f}/1.5 points on validation set!")
