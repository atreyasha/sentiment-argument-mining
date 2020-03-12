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

**Note:** R-scripts were tested with R version `3.6.*`.

**iii.** If you want to use or reference the best model in this repository (which is stored in a `git-lfs` entry), you would need to install `git-lfs` for your system.

If you already had `git-lfs` installed before cloning the repository, the best model data would be downloaded alongside the clone.

If you installed `git-lfs` after cloning this repository, execute `git lfs pull` in order to recover the large model data, as per suggestions [here](https://github.com/git-lfs/git-lfs/issues/325).

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

For training, we use the base version 2 of [ALBERT](https://github.com/google-research/ALBERT) and fine-tune it on the US Election Debate corpus in the form of a sequence tagging task. Within this task, each token of the US Election Debate corpus must be classified into one of three argument candidates; specifically "None" (N), "Claim" (C) or "Premise" (P). This is very similar to a Natural Entity Recognition (NER) task. This workflow is encompassed in `train_USElectionDebates.py`.

```
$ python3 train_USElectionDebates.py --help

usage: train_USElectionDebates.py [-h] [--max-seq-length int] [--grid-search]
                                  [--max-epochs int] [--batch-size int]
                                  [--warmup-epochs int] [--max-learn-rate float]
                                  [--end-learn-rate float] [--model-type str]
                                  [--json str]

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length int  maximum sequence length of tokenized id's (default: 512)
  --grid-search         option for grid search to be performed, otherwise single
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

arguments specific to grid-search training:
  --json str            path to json file for hyperparameter ranges (default:
                        ./utils/grid.json)
```

In our training regime, we assume a warmup-cooldown learning rate profile which entails a linear learning rate increase from `end-learn-rate` to `max-learn-rate` within the first `warmup-epochs`. Then, the learning rate exponentially decays over the remaining epochs until `max-epochs` towards `end-learn-rate`.

Furthermore, we provide 3 pre-defined simple decoder models named as `TD_Dense`, `1D_CNN` and `Stacked_LSTM`. More details on these can be seen in [model_utils.py](./utils/model_utils.py).

**i.** Under the single model training scheme, the model will be trained given a supplied set of parameters. Relevant performance histories and evaluation metrics will be stored in `./model_logs`.

An example of executing a single model training is shown below:

```shell
$ python3 train_USElectionDebates.py --model-type Stacked_LSTM --batch-size 50
```

**ii.** Under the grid-search model training scheme, models will be trained with various hyperparameters, which are defined in `./utils/grid.json`. Relevant evaluation metrics of all models and the performance history of the best model will be stored in `./model_logs`.

An example of executing a grid-search model training is shown below:

```shell
$ python3 train_USElectionDebatespy --grid-search --batch-size 30
```

**iii.** This workflow was tested on a single NVIDIA GeForce GTX 1080 Ti GPU with 12 GB RAM. Due to limited-memory issues, we had to use a low default batch-size of `10`. Our best model weights and evaluation metrics can be found in `./model_logs/2020_03_06_16_19_03_MSL512_grid_train`. Our best model with the `TD_Dense` decoder achieved a `69%` Macro-F\_1 score on the test dataset.

### 5. Visualization

In order to visualize the US Election Debate corpus and model results, we created functions in the `vis.R` script. The corresponding visualizations will be converted to `tikz` code in a latex environment and will then be saved in `./img` as `pdf` files.

```
$ Rscript vis.R --help

Usage: vis.R [options]

Options:
	-h, --help
		Show this help message and exit

	-m MODEL-DIR, --model-dir=MODEL-DIR
		Plot model evolution for specified model directory
```

**i.** In order to construct plots of the US Election Debate corpus and respective token frequencies, simply execute as follows:

```shell
$ Rscript vis.R
```

**ii.** In order to construct a plot of a model's performance history, simply append the path of the model log file as shown below:

```shell
$ Rscript vis.R --model-dir ./model_logs/2020_03_06_16_19_03_MSL512_grid_train
```

### 6. Acknowledgments

**@kpe** for BERT/ALBERT code in [bert-for-tf2](https://github.com/kpe/bert-for-tf2)
