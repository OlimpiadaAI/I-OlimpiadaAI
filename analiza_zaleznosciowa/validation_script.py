from pathlib import Path
import subprocess
import re

import torch
from utils import read_conll, uuas_score
from transformers import AutoModel, AutoTokenizer

import argparse


parser = argparse.ArgumentParser(description='Run the script with or without training.')
parser.add_argument('--train', action='store_true', help='Enable training mode.')
args = parser.parse_args()
TRAIN = args.train

python_script_filename = "zadanie_jako_skrypt.py"
DEPTH_MODEL_PATH = 'depth_model.pth'
DISTANCE_MODEL_PATH = 'distance_model.pth'


# Command to convert .ipynb to .py
command = [
    "jupyter", "nbconvert", 
    "--to", "script", 
    "--output", python_script_filename.split(".")[0], 
    "analiza_zaleznosciowa.ipynb"
]

subprocess.run(command)

if not TRAIN:
    # replace line FINAL_EVALUATION_MODE = False with FINAL_EVALUATION_MODE = True
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


if not TRAIN:
    # load function `evaluate_model`
    parse_sentence_fn = vars.get("parse_sentence", None)

    def score(root_placement, uuas):
        def scale(x, lower=0.5, upper=0.85):
            scaled = min(max(x, lower), upper)
            return (scaled - lower) / (upper - lower)
        return (scale(root_placement) + scale(uuas)) / 2

    def evaluate_model(sentences, distance_model, depth_model, tokenizer, model):
        num = 0
        sum_uuas = 0
        root_correct = 0
        with torch.no_grad():
            for sent in sentences:
                parsed = parse_sentence_fn(sent, distance_model, depth_model, tokenizer, model)
                uuas_ = uuas_score(sent, parsed)
                root_correct += int(parsed.root == sent.root)
                sum_uuas += uuas_
                num += 1
        
        root_placement = root_correct / len(sentences)
        uuas = sum_uuas / len(sentences)

        print(f"UUAS: {uuas * 100:.3}%")
        print(f"Root placement: {root_placement * 100:.3}%")
        print(f"Your score: {score(root_placement, uuas) * 100:.3}%")

    # load test set
    test_sentences = read_conll("valid.conll")

    # load models
    distance_model_class = vars.get("DistanceModel", None)
    depth_model_class = vars.get("DepthModel", None)

    distance_model_loaded = distance_model_class()
    distance_model_loaded.load_state_dict(torch.load(DISTANCE_MODEL_PATH))

    depth_model_loaded = depth_model_class()
    depth_model_loaded.load_state_dict(torch.load(DEPTH_MODEL_PATH))

    # load BERT and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")

    evaluate_model(test_sentences, distance_model_loaded, depth_model_loaded, tokenizer, model)

