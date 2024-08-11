# 1st Polish AI Olympiad

[![pl](https://img.shields.io/badge/lang-pl-red.svg)](https://github.com/OlimpiadaAI/I-OlimpiadaAI/blob/master/README.pl.md)

Welcome to the repository for the 1st Polish Artificial Intelligence Olympiad. This Olympiad is aimed at high school students in Poland who are interested in artificial intelligence. The goal is to increase interest in AI and to select a team for the [International Olympiad in Artificial Intelligence](https://ioai-official.org/).

<p align="center">
  <img src="https://raw.githubusercontent.com/OlimpiadaAI/I-OlimpiadaAI/main/logo_ioai.png" width="40%">
</p>

## General Information

**Homepage:** [Polish Artificial Intelligence Olympiad](https://oai.cs.uni.wroc.pl/english)

The 1st edition of the Olympiad was conducted in two stages. The first stage was held from April 22nd to May 27th, and consisted of seven problems which participants solved at home. During the final bootcamp held from 15th to 21st June 2024, top-30 students participated in the final competition and in the implementation contest. The competition rules are available on our website.

## Task Submission

Tasks were solved independently and were submitted to the Task Committee via the Olympiad's special website. Each task specifies which files were to be submitted — most often, this is a single Jupyter Notebook file. All submissions were evaluated automatically by a script similar to the one provided in the task. Before submitting a solution, each participant checked that it passes the validation script.

## Tasks

As part of the 1st stage of the Olympiad, participants faced the following challenges:
- **Adversarial Attacks** – Attack on a convolutional neural network.
- **Imbalanced Classification** – Training a classifier on imbalanced data.
- **Dependency Parsing** – Syntax analysis of sentences using the HerBERT model.
- **Color Quantization** – Color quantization in images.
- **Object Tracking** – Tracking objects in a video sequence.
- **Pruning** – Reducing the number of weights in neural networks.
- **Riddles** – Answering questions based on a source text.

During the final competition, participants solved the following problems:
- **Ciphers** – Machine learning algorithm to break the cipher.
- **Anomaly detection** – Detection of images outside of the training set distribution.
- **Self-supervised learning** – Time series classification.

During the implementation contest **Machine translation**, participants were challenged with implementation and simplified reproduction of results from a scientific paper.

## Environment

The list of acceptable packages is in the `requirements.txt` file. Solutions were tested using Python 3.11. For the purpose of solving the tasks, we recommend creating a virtual environment:
```
python3 -m venv oai_env
source oai_env/bin/activate
pip install -r OlimpiadaAI/requirements.txt
```

## Evaluation criteria

Scores for the tasks were calculated based on the criteria provided in the task descriptions. For solutions, participants could earn a maximum of 1.0 point (Adversarial Attacks, Imbalanced Classification), 1.5 points (Object Tracking, Pruning, Puzzles, Color Quantization), or 2.0 points (Dependency Parsing). A total of 10 points could be earned in the first stage. During the final competition, all problems were valued equally.

## Licenses

The repository uses the following licensed resources:

- **Dependency Corpus** - Resource available under the GNU General Public License version 3 (GPLv3). More information can be found [here](https://zil.ipipan.waw.pl/Sk%C5%82adnica). The dataset used in the "Dependency Parsing" task constitutes a derivative work.
- **HerBERT base cased** - Model available [here](https://huggingface.co/allegro/herbert-base-cased),
```
@inproceedings{mroczkowski-etal-2021-herbert,
    title = "{H}er{BERT}: Efficiently Pretrained Transformer-based Language Model for {P}olish",
    author = "Mroczkowski, Robert  and
          Rybak, Piotr  and
          Wr{\\'o}blewska, Alina  and
          Gawlik, Ireneusz",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bsnlp-1.1",
    pages = "1--10",
}
```
- **Datasets generated using PyBullet** - Licensed under MIT, details [here](https://github.com/hebaishi/pybullet/blob/master/LICENSE).
- **Dall-E and Stable Diffusion** - Full rights for use and sale of results, more information in the [license](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).
- **Datasets generated using SCGAN** - More information at [IEEE](https://ieeexplore.ieee.org/document/8476290) and in the [GitHub repository](https://github.com/gauss-clb/SCGAN).
- During implementation contest, part of the final stage, students implemented 
```
@article{Bahdanau2014NeuralMT,
  title={Neural Machine Translation by Jointly Learning to Align and Translate},
  author={Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio},
  journal={CoRR},
  year={2014},
  volume={abs/1409.0473},
  url={https://api.semanticscholar.org/CorpusID:11212020}
}
```

## Contact

For questions or concerns, please contact us via email: [oai@cs.uni.wroc.pl](mailto:oai@cs.uni.wroc.pl).

We wish you inspiration and good luck with the tasks!

