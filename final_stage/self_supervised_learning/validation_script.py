from pathlib import Path
import subprocess
import re
import os

import torch
import numpy as np

from PIL import Image
from functools import wraps
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, \
    accuracy_score, precision_score, f1_score, recall_score

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap
# pip install nbconvert



python_script_filename = "zadanie_jako_skrypt.py"
MODEL_PATH = 'encoder.pt'
X_TRAIN_PATH = "train_x_small.pt"
Y_TRAIN_PATH = "train_y_small.pt"
X_TEST_PATH = "val_x.pt"
Y_TEST_PATH = "val_y.pt"

# Command to convert .ipynb to .py
command = [
    "jupyter", "nbconvert", 
    "--to", "script", 
    "--output", python_script_filename.split(".")[0], 
    "self_supervised.ipynb"
]

subprocess.run(command)

# replace line CLEAN_VERSION = False with CLEAN_VERSION = True
with open(python_script_filename, "r") as file:
    filedata = file.read()
    filedata = re.sub("FINAL_EVALUATION_MODE = False", "FINAL_EVALUATION_MODE = True", filedata)

with open(python_script_filename, "w") as file:
    file.write(filedata)

# execute the python file
vars = {
    'script': str(Path(python_script_filename).name),
    '__file__': str(Path(python_script_filename)),
}
exec(open(python_script_filename).read(), vars)


def evaluate_algorithm(tne):
    X_train_small = torch.load(X_TRAIN_PATH)
    y_train_small = torch.load(Y_TRAIN_PATH)
    X_test = torch.load(X_TEST_PATH)
    y_test = torch.load(Y_TEST_PATH)
    pred = tne(X_train_small, y_train_small, X_test, MODEL_PATH)
    acc = accuracy_score(y_test.cpu().numpy(), pred.cpu().numpy())
    score = min(max((acc - 0.7), 0.), 0.2) / 0.2
    print(f"Accuracy: {acc}")
    print(f"Score: {score}")

train_and_predict_fn = vars.get("finetune_and_predict", None)
evaluate_algorithm(timing(train_and_predict_fn))