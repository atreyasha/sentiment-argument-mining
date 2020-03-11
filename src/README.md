## Sentiment and Argumentation Mining (UN Security Council Speeches)

This readme will summarize our code and results in conducting sentiment analysis and argumentation mining on the UNSC corpus.

### Table of Contents

1. [Dependencies](#1-Dependencies)
2. [Repository initialization](#2-Repository-initialization)
3. [Preprocessing](#3-Preprocessing)
4. [Training and Evaluation](#4-Training-and-Evaluation)
5. [Visualization](#5-Visualization)
6. [Acknowledgments](#6-Acknowledgments)

### 1. Dependencies

**i.** In order to set up this repository, we would need to satisfy local pythonic dependencies. If `poetry` is installed on your system, you can install dependencies and create a virtual environment automatically via the following command:

```shell
$ poetry install
```

Alternatively, you can install dependencies with `pip`:

```shell
$ pip install -r requirements.txt
```

**Note**: Your python version must be `3.7.*` in order to install certain dependencies in this repository. 

**ii.** In this repository, we use `R` and `ggplot` for visualization. Execute the following within your R console to get the dependencies:

```r
> install.packages(c("ggplot2","tikzDevice","reshape2","optparse","ggsci"))
```

### 2. Repository initialization

In order to initialize this repository, simply run `init.sh` as shown below:

```shell
$ ./init.sh
```

**i.** Firstly, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

**ii.** Secondly, you will be prompted to download and deploy the UNSC and US election debate corpus. This will download and unzip the corresponding files.

### 3. Preprocessing

For the training of the argumentation classifier model (which uses [ALBERT](https://github.com/google-research/ALBERT) for the encoder segment), we must perform significant pre-processing on the US Election Debate corpus. This includes character span conversion to token tags, `ALBERT` tokenization, addition of special `ALBERT` tokens and corpus pruning. For this, we have created the script `pre_process_USElectionDebates.py` with dedicated functions.

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

### 4. Training and Evaluation

For training, we use the base version 2 of `ALBERT` and fine-tune it on the US Election Debate corpus in the form of a sequence tagging task. Within this task, each token of the US Election Debate corpus must be classified into one of three argument candidates; specifically "None" (N), "Claim" (C) or "Premise" (P). This is very similar to a Natural Entity Recognition (NER) task. This workflow is encompassed in `train.py`.

```
$ python3 train.py --help

usage: train.py [-h] [--max-seq-length int] [--grid-search] [--max-epochs int]
                [--batch-size int] [--warmup-epochs int] [--max-learn-rate float]
                [--end-learn-rate float] [--model-type str]

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length int  maximum sequence length of tokenized id's (default: 512)
  --grid-search         True if grid search should be performed, otherwise single
                        training session will commence (default: False)
  --max-epochs int      maximum number of training epochs (default: 100)
  --batch-size int      batch-size for training procedure (default: 10)

arguments specific to single training:
  --warmup-epochs int   warmup or increasing learning rate epochs (default: 20)
  --max-learn-rate float
                        peak learning rate before exponential decay (default: 1e-05)
  --end-learn-rate float
                        final learning rate at end of planned training (default:
                        1e-06)
  --model-type str      top layer after albert, options are 'TD_Dense', '1D_CNN' or
                        'Stacked_LSTM' (default: TD_Dense)
```

In our training regime, we assume a warmup-cooldown learning rate profile which entails a linear learning rate increase frrom `end-learn-rate` to `max-learn-rate` within the first `warmup-epochs`. Then, the learning rate exponentially decays over the remaining epochs until `max-epochs` towards `end-learn-rate`.

Furthermore, we provide 3 pre-defined simple decoder models named as `TD_Dense`, `1D_CNN` and `Stacked_LSTM`. More details on these can be seen in [model_utils.py](./utils/model_utils.py).

**i.** Under this training scheme, a user can run both single and grid-search model trainings. Under the single model training scheme, the model will be trained given a suppled set of parameters. Relevant performance histories and evaluation metrics will be stored in `./model_logs`.

An example of executing a single model training is shown below:

```shell
$ python3 train.py --model-type Stacked_LSTM --batch-size 50
```

**ii.** Under the grid-search model training scheme, models will be trained with various hyperparameters, which are defined as in the `grid` dictionary in `train.py`. Relevant performance histories and evaluation metrics of the best performing model (on the test set) will be stored in `./model_logs`.

An example of executing a grid-search model training is shown below:

```shell
$ python3 train.py --grid-search --batch-size 50
```

**Note:** In order to modify the grid-search hyperparameters, the user would have to edit the `grid` variable in `train.py`.

### 5. Visualization

### 6. Acknowledgments

**@kpe** for BERT/ALBERT code in [bert-for-tf2](https://github.com/kpe/bert-for-tf2)
