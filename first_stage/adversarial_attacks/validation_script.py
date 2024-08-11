from pathlib import Path
import subprocess
import re
import os

import torch
import numpy as np

from PIL import Image


# pip install nbconvert


python_script_filename = "zadanie_jako_skrypt.py"


# Command to convert .ipynb to .py
command = [
    "jupyter", "nbconvert", 
    "--to", "script", 
    "--output", python_script_filename.split(".")[0], 
    "ataki_adwersarialne.ipynb"
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


################ DataLoader #############################################
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim

class ContestDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = Image.fromarray(self.data[index])
        if self.transform:
            x = self.transform(x)
        y = self.labels[index]
        return x, y
    
############### KOD PUNKTUJĄCY ####################################
######################### NIE ZMIENIAJ TEJ KOMÓRKI ##########################
def evaluate_network(data_loader,
                     model,
                     device,
                     verbose=True):
    # Dokonajmy ewaluacji wytrenowanego modelu na wybranym zbiorze
    all_predictions, all_labels = [], []
    model.to(device)
    # Sieć neuronowa musi zostać przeniesiona do trybu ewaluacji
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Wybieramy klasę o najwyższym prawdopodobieństwie przynależności
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted.ravel())
            all_labels.append(labels.ravel())
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # Sprawdzamy, ile etykiet zostało prawidłowo wytypowanych przez sieć
    correct = (all_predictions == all_labels).sum().item()
    accuracy = (100 * correct / all_labels.size()[0])
    no_of_elements = len(data_loader.dataset)
    if verbose:
        print(f'Dokładność klasyfikacji na {no_of_elements} obrazów wybranego zbioru wynosi '
              f'{accuracy} %.')
    return accuracy

def calculate_similarity(original_dataset,
                         perturbed_dataset):
    # Zarówno original_dataset jak i perturbed_dataset mają być typu Numpy array.
    # Mają mieć taki sam rozmiar, tj. (liczba elementów x 28 x 28)
    assert original_dataset.shape == perturbed_dataset.shape
    assert original_dataset.shape[1] == original_dataset.shape[2] == 28
    assert perturbed_dataset.shape[1] == perturbed_dataset.shape[2] == 28
    similarities, L1_distances = [], []
    for i in range(original_dataset.shape[0]):
        original_element = original_dataset[i].ravel()
        perturbed_element = perturbed_dataset[i].ravel()
        similarities.append(
            ssim(
                original_element,
                perturbed_element,
                data_range=2)
            )
        L1_distances.append(
            np.max(np.abs(
                original_element - perturbed_element
            ))
        )
    mean_SSIM = np.mean(similarities)
    max_distance = np.max(L1_distances)
    print(f'Średnia wartość SSIM wynosi {mean_SSIM}, a największa odległość między pikselami wynosi: {max_distance}.')
    return mean_SSIM, max_distance
###############################################################################



def evaluate_algorithm(model, images, labels, device, perturbe_algorithm):
    dataset_for_perturbation = deepcopy(images)
    dataset_attacked = perturbe_algorithm(dataset_for_perturbation)

    SSIM, distance = calculate_similarity(
        dataset_for_perturbation,
        dataset_attacked
    )
    assert distance <= 0.3

    perturbed_set = ContestDataset(dataset_attacked,
                                labels,
                                transform=transforms.ToTensor())
    perturbed_loader = DataLoader(perturbed_set,
                                batch_size=64,
                                shuffle=False)
    perturbed_accuracy = evaluate_network(
        perturbed_loader,
        model,
        device,
        verbose=True
    )

perturbe_dataset_fn = vars.get("perturbe_dataset", None)
dataset = vars.get("X_validation", None)
labels = vars.get("y_validation", None)
model = vars.get("net", None)
device = vars.get("device", None)

evaluate_algorithm(model, dataset, labels, device, perturbe_dataset_fn)

