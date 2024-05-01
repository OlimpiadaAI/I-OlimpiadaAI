from pathlib import Path
import subprocess
import re
import os

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# Możesz zmienić tę flagę, żeby wytrenować swój model
TRAIN = False

NOTEBOOK_AS_SCRIPT = "zadanie_jako_skrypt.py"
MODEL_PATH = "cnn-classifier.pth"

NOTEBOOK_NAME = "niezbalansowana_klasyfikacja.ipynb"

######################### KOD Z NOTEBOOKA ##########################

class ImageDataset(torch.utils.data.Dataset):
    """Implementacja abstrakcji zbioru danych z torch'a."""
    def __init__(self, dataset_type: str):
        self.filelist = glob.glob(f"{dataset_type}_data/*")
        self.labels   = [0 if "normal" in path else 1 for path in self.filelist]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = torchvision.transforms.functional.to_tensor(plt.imread(self.filelist[idx])[:,:,0])
        label = self.labels[idx]
        return image, label
    
    def loader(self, **kwargs) -> torch.utils.data.DataLoader:
        """
        Stwórz, `DataLoader`'a dla aktualnego zbioru danych.

        Wszystkie `**kwargs` zostaną przekazane do konstruktora `torch.utils.data.DataLoader`.
        `DataLoader`'y w skrócie to abstrakcja ładowania danych usdostępniająca wygodny interfejs.
        Możesz dowiedzieć się o nich więcej tutaj: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """
        return torch.utils.data.DataLoader(self, **kwargs)
    
if not os.path.exists("valid_data"):
    exit("Nie znaleziono zbioru testowego. Uruchom notebook'a w celu jego pobrania.")

valid_dataset: ImageDataset = ImageDataset("valid")

def accuracy_to_points(accuracy: float) -> float:
    """Oblicz wynik na podstawie celności predykcji."""
    return (round(accuracy, 2) - 0.5) * 2 if accuracy > 0.5 else 0.0

def grade(model):
    """Oceń ile punktów otrzyma aktualne zadanie."""
    model.eval()
    model

    test_loader = valid_dataset.loader()
    correct = 0
    total = 0
    with torch.no_grad():
        for [images, labels] in test_loader:
            outputs = model(images).squeeze()
            incorrect_indices = torch.where((outputs > 0.5).int() != labels)[0]
            correct += len(labels) - len(incorrect_indices)
            total += len(labels)
        accuracy = correct / total if total != 0 else 0
        print(f"Accuracy: {int(round(accuracy, 2) * 100)}%")
        return accuracy_to_points(accuracy)

def evaluate_model(model):
    """Oceń ile punktów otrzyma aktualne zadanie."""
    return grade(model)

def main():
    
    command = [
        "jupyter", "nbconvert", 
        "--to", "script", 
        "--output", NOTEBOOK_AS_SCRIPT.split(".")[0], 
        NOTEBOOK_NAME
    ]

    subprocess.run(command)

    if not os.path.exists(MODEL_PATH):
        exit(f"Błąd: wymagany plik '{MODEL_PATH}' z parametrami modelu nie znaleziony w {os.getcwd()}")

    if not TRAIN:
        with open(NOTEBOOK_AS_SCRIPT, "r", encoding="utf8") as file:
            filedata = file.read()
            filedata = re.sub("FINAL_EVALUATION_MODE = False", "FINAL_EVALUATION_MODE = True", filedata)

        with open(NOTEBOOK_AS_SCRIPT, "w") as file:
            file.write(filedata)

    vars = {
        'script': str(Path(NOTEBOOK_AS_SCRIPT).name),
        '__file__': str(Path(NOTEBOOK_AS_SCRIPT)),
    }
    exec(open(NOTEBOOK_AS_SCRIPT).read(), vars)

    YourCnnClassifier = vars.get("YourCnnClassifier", None)

    if YourCnnClassifier is None:
        exit(f"Błąd: Wymagana klasa 'YourCnnClassifier' nie znajduje się w '{NOTEBOOK_NAME}'.")
    
    if not hasattr(YourCnnClassifier, "load"):
        exit(f"Błąd: wymagana metoda 'load' klasy 'YourCnnClassifier' nie znajduje się w '{NOTEBOOK_NAME}'.")

    model = YourCnnClassifier.load()

    point_grade = evaluate_model(model)
    print(f"Ocena: {point_grade} pkt")

if __name__ == "__main__":
    main()
