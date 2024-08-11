from pathlib import Path
import subprocess
import re
import os


python_script_filename = "zadanie_jako_skrypt.py"

# Command to convert .ipynb to .py
command = [
    "jupyter", "nbconvert", 
    "--to", "python", 
    "--output", python_script_filename.split(".")[0], 
    "szyfry.ipynb"
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

path_to_data = ''

clear_file_path = os.path.join(path_to_data, "clear_lines.txt")
ciphered_file_path = os.path.join(path_to_data, "ciphered_lines.txt")
solutions_file_path = os.path.join(path_to_data, "ciphered_lines_ground_truth.txt")

corpus_clear = [line.strip().lower() for line in open(clear_file_path)]
corpus_ciphered = [line.strip().lower() for line in open(ciphered_file_path)]
corpus_original = [line.rstrip('\n').lower() for line in open(solutions_file_path)]

############### KOD PUNKTUJĄCY ####################################

def accuracy_metric(original_lines, deciphered_lines):
    original_str = "".join(original_lines)
    deciphered_str = "".join(deciphered_lines)
    assert len(original_str) == len(deciphered_str)
    good_char = sum(int(a == b) for a, b in zip(original_str, deciphered_str))
    return good_char / len(original_str)


def evaluate_algorithm(clear_data, original_data, ciphered_data):
    decipher_corpus_fn = vars.get("decipher_corpus", None)
    deciphered = decipher_corpus_fn(corpus_clear, corpus_ciphered)
    accuracy = accuracy_metric(corpus_original, deciphered)
    score = 2 * max(accuracy - 0.5, 0.0)
    print("-- start of validation script output --")
    print(f"Accuracy: {accuracy}")
    print(f'Twój wynik to {score} pkt')
    print("-- end of validation script output --")
    return score


evaluate_algorithm(corpus_clear, corpus_original, corpus_ciphered)

