## Sentiment and Argumentation Mining (UN Security Council Speeches)

This readme will summarize our code and results in conducting sentiment analysis and argumentation mining on the UNSC corpus.

### Table of Contents

1. [Dependencies](#1-Dependencies)
2. [Repository initialization](#2-Repository-initialization)
3. [Preprocessing](#3-Preprocessing)
4. [Training](#4-Training/Evaluation)
5. [Visualization](#5-Visualization)
6. [Acknowledgments](#6-Acknowledgments)

### 1. Dependencies

1. In order to set up this repository, we would need to satisfy local pythonic dependencies. If `poetry` is installed on your system, you can install dependencies and create a virtual environment automatically via the following command:

```shell
$ poetry install
```

Alternatively, you can install dependencies with `pip`:

```shell
$ pip install -r requirements.txt
```

**Note**: Your python version must be `3.7.*` in order to install certain dependencies in this repository. 

2. In this repository, we use `R` and `ggplot` for visualization. Execute the following within your R console to get the dependencies:

```r
> install.packages(c("ggplot2","tikzDevice","reshape2","optparse","ggsci"))
```

### 2. Repository initialization

In order to initialize this repository, simply run `init.sh` as shown below:

```shell
$ ./init.sh
```

1. Firstly, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

2. Secondly, you will be prompted to download and deploy the UNSC and US election debate corpus. This will download and unzip the corresponding files.

### 3. Preprocessing

For the training of the argumentation classifier model (which uses [ALBERT](https://github.com/google-research/ALBERT) for the encoder segment), we must perform significant pre-processing on the US Election Debate corpus. For this, we have created the script `pre_process_USElectionDebates.py` with dedicated functions.

```
$ python3 pre_process_USElectionDebates.py --help

usage: pre_process_USElectionDebates.py [-h] [--max-seq-length int]

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length int  maximum sequence length of tokenized id's (default: 512)
```

In our training process, we assume the maximum possible sequence length of `512` tokens for the `ALBERT` encoder model. In order to conduct pre-processing, simply execute the following:

```shell
$ python3 pre_process_USElectionDebates.py
```

This process will produce respective `json`, `csv` and `npy` files in the `./data` directory; all of which will be later utilized in training and evaluation.

### 4. Training/Evaluation

### 5. Visualization

### 6. Acknowledgments

**@kpe** for BERT/ALBERT code in [bert-for-tf2](https://github.com/kpe/bert-for-tf2)
