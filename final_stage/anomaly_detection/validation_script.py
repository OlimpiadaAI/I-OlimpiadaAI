import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import re
import threading
import time

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import typing as t
import logging

from pathlib import Path

import pandas as pd
import torch


# ============== Customizacja wyświetlania ============== #

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def error(message: str, hints: list[str] | None = None):
    logging.critical("[💀] " + message)
    if hints is not None:
        for hint in hints:
            logging.info(" - [💡] " + hint)
    os._exit(1)

def ok(message: str):
    logging.info("[✅] " + message)


# ============== Configuracja sprawdzaczki ============== #

parser = argparse.ArgumentParser(description="Sprawdzaczka")
parser.add_argument(
    "--notebook", 
    type=str, 
    default="anomaly_detection.ipynb", 
    help="Ścieżka do notebook'a"
)
parser.add_argument(
    "--script-out",
    type=str,
    default="zadanie_jako_skrypt.py",
    help="Ścieżka gdzie `nbconvert` ma wypluć skrypt z notebook'a"
)

args = parser.parse_args()

SCRIPT_NAME: str = args.script_out
NOTEBOOK_NAME: str = args.notebook
TIME_BUDGET_IN_SECONDS: int = 15 * 60
REQUIRED_NOTEBOOK_ARTIFACTS: list[str] = ["BATCH_SIZE", "model"]


# ============== Przygotowanie skryptu z notebook'a ============== #

command = [
    "jupyter", "nbconvert", 
    "--to", "script", 
    "--output", SCRIPT_NAME.split(".")[0], 
    NOTEBOOK_NAME
]

subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

with open(SCRIPT_NAME, "r", encoding="utf-8") as file:
    filedata = file.read()
    filedata = re.sub("FINAL_EVALUATION_MODE = False", "FINAL_EVALUATION_MODE = True", filedata)

with open(SCRIPT_NAME, "w", encoding="utf-8") as file:
    file.write(filedata)

# Function to execute the Python script
notebook_defines: dict[str, t.Any] = {
    'script': str(Path(SCRIPT_NAME).name),
    '__file__': str(Path(SCRIPT_NAME)),
}
def exec_script():
    exec(open(SCRIPT_NAME, encoding="utf-8").read(), notebook_defines)


# ============== Trenowanie modelu ============== #

train_start = time.time()

# Create a thread to run the exec_script function
thread = threading.Thread(target=exec_script)

# Start the thread
thread.start()

# Wait for the specified timeout
thread.join(float(TIME_BUDGET_IN_SECONDS))
train_end = time.time()
training_time = train_end - train_start
time_elapsed = training_time
remaining_time = TIME_BUDGET_IN_SECONDS - time_elapsed

if thread.is_alive():
    error(f"Trenowanie modelu z `{NOTEBOOK_NAME}` przekroczyło limit czasowy {TIME_BUDGET_IN_SECONDS} sekund.")

ok(f"Trenowanie modelu ukończone w {time_elapsed:.2f} sekund -- zostało {remaining_time:.2f} sekund na testy.")


# ============== Weryfikacja struktury notebook'a ============== #

for variable in REQUIRED_NOTEBOOK_ARTIFACTS:
    if variable not in notebook_defines:
        error(
            f"Nie znaleziono globalnej definicji zmiennej `{variable}` w notebook'u `{NOTEBOOK_NAME}`.", 
            hints=[
                "Upewnij się, że nie jest ona wykomentowana lub zdefiniowana w ciele funkcji."
            ]
        )

ok("Wszystkie wymagane zmienne znalezione.")


# ============== Przygotowanie środowiska testowego ============== #

BATCH_SIZE: int = int(notebook_defines["BATCH_SIZE"])
model = notebook_defines["model"]

TRAIN_DIR: Path = Path("./train")
VALID_DIR: Path = Path("./valid")

TRAIN_CSV: Path = Path("./train.csv")
VALID_CSV: Path = Path("./valid.csv")

class ImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, dir: Path, csv: Path):
        self.dir: Path = dir
        self.csv = pd.read_csv(csv)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path, label = self.csv.iloc[idx]
        img = plt.imread(self.dir / path)

        return transforms.functional.to_tensor(img), label

def test_dataloader() -> DataLoader:
    return DataLoader(ImageDataset(VALID_DIR, VALID_CSV), batch_size=BATCH_SIZE, shuffle=True)

def grade_solution(model):
    dataloader = valid_dataloader()

    predictions = np.concatenate([
        model.predict(images).cpu().to(dtype=torch.int32).numpy() for images, _ in dataloader
    ], axis=0)

    labels = np.concatenate([label for _, label in dataloader], axis=0, dtype=np.int32)

    accuracy = sum(labels == predictions) / len(labels)
    
    score = min(max(accuracy - 0.6, 0.0), 0.3) / 0.3

    print(f"Accuracy: {accuracy}")
    print(f'Twój wynik to {score} pkt')
    return score


# ============== Weryfikacja struktury modelu ============== #

if not hasattr(model, "predict"):
    error("Nie znaleziono metody `predict` w modelu.")

predict_method = inspect.signature(model.predict)
parameters = predict_method.parameters
if len(parameters) != 1:
    expected_signature = "def predict(self, batch):"
    actual_signature = f"def predict(self, {', '.join(parameters.keys())}):"
    
    error(f"Zła ilość argumentów w metodzie `predict`\n - oczekiwano 2: `{expected_signature}`\n - otrzymano {len(parameters.keys())}: `{actual_signature}`")

ok("Znaleziono metodę `predict` o poprawnej sygnaturze.")


# ============== Testy modelu ============== #

test_start = time.time()
score = grade_solution(model)
test_end = time.time()
testing_time = test_end - test_start

time_elapsed += testing_time

if time_elapsed >= TIME_BUDGET_IN_SECONDS:
    error(f"Testowanie modelu przekroczyło limit czasowy {remaining_time:.2f} sekund.")

ok(f"Testowanie modelu zakończone w {testing_time:.2f} sekund -- zapas {TIME_BUDGET_IN_SECONDS - time_elapsed:.2f} sekund.")
