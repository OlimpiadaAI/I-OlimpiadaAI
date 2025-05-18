import os
from PIL import Image
import json
import shutil
import zipfile
import gdown

def download_and_replace_data(train=("1OXfB1bwnOf8hHXA5Affz_feAjja_99MG", "train_data.zip"),
                              valid=("1ErNoW6Xa3ItCCGAiB-UCwq1Bx9l89Iwf", "valid_data.zip")):
    GDRIVE_DATA = [train, valid]
    for file_id, zip_name in GDRIVE_DATA:
        folder_name = zip_name.split(".")[0]  # Assuming folder name is the same as the file name without '.zip'
        
        # Remove existing folder if it exists
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
            print(f"Removed existing directory: {folder_name}")

        # Download zip file
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, zip_name, quiet=False)
        
        # Extract zip file
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall()  # Extract files into the current directory
        os.remove(zip_name)  # Remove the zip file after extraction
        print(f"Extracted and cleaned up {zip_name}")


def _load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, filename)).convert('RGBA')
        images.append(img)
    return images


def get_level_info(level=1,dataset="train"):
    """
    Zwraca liczbę filmów i klatek na danym poziomie.

    Parametry:
    level (int): Poziom, dla którego należy pobrać informacje.
    dataset (str): Nazwa zbioru danych, "train" lub "valid".

    Zwraca:
    tuple: Krotka zawierająca liczbę filmów i listę liczby klatek w każdym filmie.
    """
    path_to_images = os.path.join(os.getcwd(), f"{dataset}_data/level_{level}/images")
    videos = [f for f in os.listdir(path_to_images)]
    num_videos = len(videos)
    num_frames = [len(os.listdir(os.path.join(path_to_images, video))) for video in videos]
    return num_videos, num_frames


def get_video_data(level=1, video_id=0, dataset="train"):
    """
    Ładuje obrazy, współrzędne i cel dla danego wideo.

    Parametry:
    level (int): Poziom zbioru danych.
    video_id (int): Identyfikator wideo do załadowania.
    dataset (str): Nazwa zbioru danych, "train" lub "valid".


    Zwraca:
    tuple: Krotka zawierająca obrazy, współrzędne, cel oraz ścieżkę do obrazów.
           Gdzie obrazy to lista obiektów PIL Image, współrzędne to słownik zawierający współrzędne obiektów w każdej klatce,
           np. {"frame_0001.jpg": [[857.2730712890625, 443.7888488769531, 1123.3863525390625, 672.2659301757812], [284.8743896484375, 432.0739440917969, 523.8243408203125, 644.8782958984375], [573.1364135742188, 438.8981628417969, 814.3084716796875, 665.8314819335938]], "frame_0002.jpg": ...}
           odpowiadające współrzędnym "X_min", "Y_min", "X_max", "Y_max" prostokątów ograniczających obiekty w każdej klatce.
           Cel to lista zawierająca prawdziwą permutację pokazaną na wideo.
           path_to_images to ciąg znaków zawierający ścieżkę do folderu z obrazami.
    """

    path_to_images = os.path.join(os.getcwd(), f"{dataset}_data/level_{level}/images/animation_{str(video_id).zfill(4)}")
    images = _load_images_from_folder(path_to_images)

    # check if the file exists
    if os.path.exists(
            os.path.join(os.getcwd(), f"{dataset}_data/level_{level}/coordinates/coordinates_{str(video_id).zfill(4)}.json")):
        with open(os.path.join(os.getcwd(),
                               f"{dataset}_data/level_{level}/coordinates/coordinates_{str(video_id).zfill(4)}.json"),
                  'r') as f:
            coordinates = json.load(f)
    else:
        coordinates = {}

    with open(os.path.join(os.getcwd(), f"{dataset}_data/level_{level}/target/target_{str(video_id).zfill(4)}.json"),
              'r') as f:
        target = json.load(f)

    return images, coordinates, target, path_to_images
