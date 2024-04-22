from pathlib import Path
import subprocess
import re
import os
import sys
import time
import numpy as np

from utils.utils import get_level_info, get_video_data

def submission_script_task_1_2(algorithm,level,verbose=False,dataset="valid"):
    num_videos, _ = get_level_info(level=level,dataset=dataset)
    correct = []
    exception_messages = set()
    for video_number in range(num_videos):
        _, coordinates, target, _ = get_video_data(level=level,video_id=video_number,dataset=dataset)
        try:
            prediction = algorithm(coordinates)
            if tuple(target) == tuple(prediction):
                correct.append(1)
            else:
                correct.append(0)
            if verbose:
                print(f"Video: animation_{str(video_number).zfill(4)}")
                print(f"Prediction: {prediction}")
                print(f"Target:     {target}")
                print(f"Score: {tuple(target) == tuple(prediction)}", end='\n\n')
        except Exception as e:
            correct.append(0)
            exception_messages.add(str(e))
    if verbose:
        print(f"Accuracy: {np.mean(correct)}")
        print(f"Correctness: {correct}")
    return np.sum(correct) / num_videos, correct, exception_messages


def submission_script_task_3(algorithm,level,verbose=False,dataset="valid"):
    num_videos, _ = get_level_info(level=level,dataset=dataset)
    correct = []
    exception_messages = set()
    for video_number in range(num_videos):
        images, _, target, _ = get_video_data(level=level,video_id=video_number,dataset=dataset)
        try:
            prediction = algorithm(images)
            if tuple(target) == tuple(prediction):
                correct.append(1)
            else:
                correct.append(0)
            if verbose:
                print(f"Video: animation_{str(video_number).zfill(4)}")
                print(f"Prediction: {prediction}")
                print(f"Target:     {target}")
                print(f"Score: {tuple(target) == tuple(prediction)}", end='\n\n')
        except Exception as e:
            correct.append(0)
            exception_messages.add(str(e))
    if verbose:
        print(f"Accuracy: {np.mean(correct)}")
        print(f"Correctness: {correct}")
    return np.sum(correct) / num_videos, correct, exception_messages

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
object_tracking_dir = os.path.join(parent_dir, 'sledzenie_obiektow')

# changes current working directory to object_tracking in order to match cwd of a notebook
os.chdir(object_tracking_dir)

from utils.utils import get_level_info, get_video_data, display_video

DATASET = "valid"

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        return True

for subtask_number in range(1,4):
    python_script_filename = f"per_video_detection_{subtask_number}_SCRIPT.py"

    # Command to convert .ipynb to .py
    command = [
        "jupyter", "nbconvert",
        "--to", "script",
        "--output", python_script_filename.split(".")[0],
        f"sledzenie_obiektow_{subtask_number}.ipynb"
    ]


    result = subprocess.run(command, capture_output=True, text=True)


    with open(python_script_filename, "r", encoding="UTF-8") as file:
        filedata = file.read()
        filedata = re.sub("FINAL_EVALUATION_MODE = False", "FINAL_EVALUATION_MODE = True", filedata)
        filedata = re.sub(r"^#.*\n", "", filedata, flags=re.MULTILINE)

    with open(python_script_filename, "w", encoding="UTF-8") as file:
        file.write(filedata)

    # execute the python file
    vars = {
        'script': str(Path(python_script_filename).name),
        '__file__': str(Path(python_script_filename)),
    }

    start_time = time.time()
    with SuppressPrint():
        exec(open(python_script_filename).read(), vars)
    end_time = time.time()

    execution_time = (end_time - start_time) / 60
    print(f"Execution time task {subtask_number}: {execution_time} minutes")

    #  retrieve functions from the executed script
    your_algorithm_task = vars.get(f"your_algorithm_task_{subtask_number}", None)
    submission_script = vars.get("submission_script", None)
    raport = vars.get(f"raport_{subtask_number}", "None")

    
    # test the algorithms
    start_time = time.time()
    if subtask_number == 3:
        accuracy, _, _ = submission_script_task_3(
            algorithm=your_algorithm_task,
            level=subtask_number,
            dataset=DATASET)
    else:
        accuracy, _, _ = submission_script_task_1_2(
            algorithm=your_algorithm_task,
            level=subtask_number,
            dataset=DATASET)
    print(f"Task {subtask_number}: {accuracy}")

    end_time = time.time()
    testing_time = (end_time - start_time) / 60
    print(f"Testing time: {testing_time} minutes")
    print(f"Total time: {execution_time + testing_time} minutes")

    print()
    print(f"Raport {subtask_number}: {raport}\nWord count: {len(raport.split())}\n")

    # cleanup
    os.remove(python_script_filename)
