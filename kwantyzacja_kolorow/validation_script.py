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
    "kwantyzacja_kolorow.ipynb"
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
import glob
from typing import Iterator
from PIL import Image
import numpy as np

class ImageDataset:
    """Klasa, która ułatwia wczytywanie obrazków z danego folderu."""
    def __init__(self, image_dir: str):
        self.filelist = glob.glob(image_dir + "/*.jpg")
        self.IMAGE_DIMS = (512,512)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx) -> np.ndarray:
        with Image.open(self.filelist[idx]) as image:
            image = image.convert('RGB')
            image = image.resize(self.IMAGE_DIMS)
            return np.array(image)

    def __iter__(self) -> Iterator[np.ndarray]:
        return (self[i] for i in range(len(self.filelist)))


############### KOD PUNKTUJĄCY ####################################
######################### NIE ZMIENIAJ TEJ KOMÓRKI ##########################

# Poniżej znajdziesz definicje MSE oraz kosztu użycia kolorów
# Pamiętaj, żeby przy ewaluacji liczyć je w przestrzeni RGB, tzn. na wartościach całkowitych z przdziału [0, 255]
# Skalowanie jest dopuszczalne tylko podczas treningu!

# Zdefiniujmy kryterium oceny jakości kwantyzacji
# Użyjemy do tego błędu średniokwadratowego (mean square error - MSE)
def mse(img, img_quant):
  return ((img_quant.astype(np.float32) - img.astype(np.float32))**2).mean()


# Następnie zdefinujmy koszt użycia kolorów
# Im bliżej danemu kolorowi do "prostych" kolorów, tym mniejszy koszt jego użycia
def color_cost(img_quant):
    vertices = np.array([
        [0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255],
        [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]
    ])
    colors = np.unique(img_quant.reshape(-1,3), axis=0)
    
    differences = colors[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    squared_distances = np.sum(differences**2, axis=2)
    costs = np.sqrt(np.min(squared_distances, axis=1))

    return np.mean(costs), colors.shape[0]


# Całkowite kryterium zdefiniowane w treści zadania
def quantization_score(img, img_quant):
    assert img.dtype == np.uint8
    assert img_quant.dtype == np.uint8
    
    mse_cost = mse(img, img_quant)
    color_cost_val, colors_num = color_cost(img_quant)
    print(f'MSE: {mse_cost:.4f}, color cost: {color_cost_val:.4f}, colors: {colors_num:.4f}')
    return mse_cost / 24 + color_cost_val + colors_num / 24
###############################################################################



def evaluate_algorithm(quantization_algorithm, data_dir):
    dataset = ImageDataset(data_dir)
    scores = []
    for image in dataset:
        quantized_image = quantization_algorithm(image)
        # show_quantization_results(image, quantized_image)
        score = quantization_score(image, quantized_image)
        scores.append(score)
    return np.mean(scores)

your_quantization_algorithm_fn = vars.get("your_quantization_algorithm", None)
evaluate_algorithm(your_quantization_algorithm_fn, 'valid_data')

