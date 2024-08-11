from pathlib import Path
import subprocess
import re
import os



# pip install nbconvert


python_script_filename = "zadanie_jako_skrypt.py"


# Command to convert .ipynb to .py
command = [
    "jupyter", "nbconvert", 
    "--to", "script", 
    "--output", python_script_filename.split(".")[0], 
    "zagadki.ipynb"
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
from nltk.tokenize import word_tokenize as tokenize
from tqdm import tqdm
############### KOD PUNKTUJĄCY ####################################
######################### NIE ZMIENIAJ TEJ KOMÓRKI ##########################
def mean_reciprocal_rank(real_answers, computed_answers, K=20):
    positions = []

    for real_answer, computed_answer in zip(real_answers, computed_answers):
        if real_answer in computed_answer[:K]:
            pos = computed_answer.index(real_answer) + 1
            positions.append(1/pos)
    
    mrr = sum(positions) / len(real_answers)
    print ('Mean Reciprocal Rank =', mrr)
    
    return mrr
###############################################################################


def evaluate_algorithm(score_function, queries, answers, K):
    computed_answers = []
    for query in tqdm(queries, desc="queries answered"):
        computed_answers.append(score_function(set(query), K=K))
    score = mean_reciprocal_rank(answers, computed_answers, K=K)
    
    return score

score_fn = vars.get("answer_riddle", None)
answers = vars.get("answers", None)
queries = vars.get("queries", None)

PART_OF_DATA = 100
answers = answers[:100]
queries = queries[:100]

evaluate_algorithm(score_fn, queries, answers, K=20)

