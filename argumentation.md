## Argumentation Mining

### Train on US Election Debates, Predict on UNSC

This readme will summarize our code and results in conducting argumentation mining on the US Election Debate and UNSC corpora.

### Table of Contents

1. [Preprocessing](#1-Preprocessing)
2. [Training and Evaluation](#2-Training-and-Evaluation)
3. [Prediction on UNSC](#3-Prediction-on-UNSC)
4. [Visualization](#4-Visualization)
5. [Acknowledgments](#5-Acknowledgments)

### 1. Preprocessing

**i.** For the training of the argumentation classifier model (which uses [ALBERT](https://github.com/google-research/ALBERT) for the encoder segment), we must perform significant preprocessing on the US Election Debate [corpus](https://github.com/ElecDeb60To16/Dataset), which was detailed in [Haddadan et al. 2019](https://www.aclweb.org/anthology/P19-1463/). This includes character span conversion to token tags, `sentencepiece` tokenization, addition of special `BERT` tokens and corpus pruning. For this, we have created the script `pre_process_USElectionDebates.py` with dedicated functions.

```
$ python3 pre_process_USElectionDebates.py --help

usage: pre_process_USElectionDebates.py [-h] [--max-seq-length int]
                                        [--verbosity int]

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length int  maximum sequence length of tokenized id's (default: 512)
  --verbosity int       0 for no text, 1 for verbose text (default: 1)
```

In our training process, we assume the maximum possible sequence length of `512` tokens for the `ALBERT` encoder model. In order to conduct preprocessing, simply execute the following:

```shell
$ python3 pre_process_USElectionDebates.py
```

This process will produce respective `json`, `csv` and `npy` files in the `./data` directory; all of which will be later utilized in training and evaluation.

**ii.** A similar type of preprocessing must be conducted on the UNSC corpus, such that we can apply the aforementioned trained classifier on it. This is done through the script `pre_process_UNSC.py`.

```
$ python3 pre_process_UNSC.py --help

usage: pre_process_UNSC.py [-h] [--max-seq-length int] [--verbosity int]

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length int  maximum sequence length of tokenized id's (default: 512)
  --verbosity int       0 for no text, 1 for verbose text (default: 1)
```

In order to conduct preprocessing of the UNSC corpus, simply execute the following:

```shell
$ python3 pre_process_UNSC.py
```

This process will produce respective `json`, `csv` and `npy` files in the `./data` directory; all of which will be later utilized in prediction.

### 2. Training and Evaluation

For training, we use the base version 2 of [ALBERT](https://github.com/google-research/ALBERT) and fine-tune it on the US Election Debate corpus in the form of a sequence tagging task. Within this task, each token of the US Election Debate corpus must be classified into one of three argument candidates; specifically "None" (N), "Claim" (C) or "Premise" (P). This is very similar to a Natural Entity Recognition (NER) task. This workflow is encompassed in `train_USElectionDebates.py`.

```
$ python3 train_USElectionDebates.py --help

usage: train_USElectionDebates.py [-h] [--max-seq-length int] [--grid-search]
                                  [--max-epochs int] [--batch-size int]
                                  [--patience int] [--warmup-epochs int]
                                  [--max-learn-rate float] [--end-learn-rate float]
                                  [--model-type str] [--json str]

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length int  maximum sequence length of tokenized id's (default: 512)
  --grid-search         option for grid search to be performed, otherwise single
                        training session will commence (default: False)
  --max-epochs int      maximum number of training epochs (default: 100)
  --batch-size int      batch-size for training procedure (default: 10)
  --patience int        number of epochs where validation metric is allowed to
                        worsen before stopping training (default: 5)

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

In our training regime, we assume a warmup-cooldown learning rate profile which entails a linear learning rate increase from `max-learn-rate/warmup-epochs` to `max-learn-rate` within the first `warmup-epochs`. Then, the learning rate exponentially decays over the remaining epochs until `max-epochs` from `max-learn-rate` to `end-learn-rate`.

Furthermore, we provide 3 pre-defined simple decoder models named as `TD_Dense`, `1D_CNN` and `Stacked_LSTM`. More details on these can be seen in [model_utils.py](./utils/model_utils.py).

**i.** Under the single model training scheme, the model will be trained given a supplied set of parameters. The model, its performance history and evaluation metrics will be stored in `./model_logs`.

An example of executing a single model training is shown below:

```shell
$ python3 train_USElectionDebates.py --model-type Stacked_LSTM --batch-size 50
```

**ii.** Under the grid-search model training scheme, models will be trained with various hyperparameters, which are defined in `./utils/grid.json`. Relevant evaluation metrics of all models, as well as the the best model and its performance history will be stored in `./model_logs`.

An example of executing a grid-search model training is shown below:

```shell
$ python3 train_USElectionDebates.py --grid-search --batch-size 30
```

**iii.** This workflow was tested on a single NVIDIA GeForce GTX 1080 Ti GPU with 12 GB RAM. Due to limited-memory issues, we had to use a low default batch-size of `10`. Our best model weights and evaluation metrics can be found in `./model_logs/2020_03_17_09_17_44_MSL512_grid_train`. Our best model with the `TD_Dense` decoder achieved a `69%` Macro-F<sub>1</sub> score on the test dataset.

### 3. Prediction on UNSC

After training, we can utilize our best model to predict argumentation candidates on the UNSC corpus. To do this, we can use the script `predict_UNSC.py`.

```
$ python3 predict_UNSC.py --help

usage: predict_UNSC.py [-h] [--max-seq-length int] [--force-pred] [--verbosity int]
                       --model-dir str

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length int  maximum sequence length of tokenized id's (default: 512)
  --force-pred          option to force redoing prediction despite presence of
                        already produced binary (default: False)
  --verbosity int       0 for no text, 1 for verbose text (default: 1)

required name arguments:
  --model-dir str       path to model *h5 file (default: None)
```

This script will load the best saved model and will predict the argumentation candidates on the preprocessed `UNSC` corpus. Next, it will save various files to the `./data` folder, which can be used for analysis and visualization of results.

To run this script on our best model, simply execute:

```shell
$ python3 predict_UNSC --model-dir ./model_logs/2020_03_17_09_17_44_MSL512_grid_train/model_1.h5
```

Under the default maximum sequence length of `512` tokens, this process will produce a clean human-readable `json` file, specifically `./data/UNSC/pred/pred_clean_512.json`, where each key maps to a speech id and the value of the key represents the tokenized version of the speech along with token-based argumentation classifications. 

This `json` file already comes pre-packaged within this repository. This file can be used for semantic analyses on how the classifier performed on the UNSC corpus.

### 4. Visualization

In order to visualize the US Election Debate and UNSC corpora, as well as model results; we created functions in the `vis.R` script. The corresponding visualizations will be converted to `tikz` code in a latex environment and will then be saved in `./img` as `pdf` files.

```
$ Rscript vis.R --help

Usage: vis.R [options]

Options:
	-h, --help
		Show this help message and exit

	-m MODEL-HISTORY, --model-history=MODEL-HISTORY
		Plot model evolution for specified model history csv file

	-p PREDICTIONS, --predictions=PREDICTIONS
		Plot prediction token distribution for given csv file
```

**i.** In order to construct plots of the US Election Debate and UNSC corpora with respective token frequencies, simply execute as follows:

```shell
$ Rscript vis.R
```

**ii.** In order to construct a plot of a model's performance history, simply append the path of the model history `csv` file as shown below:

```shell
$ Rscript vis.R --model-history \
./model_logs/2020_03_17_09_17_44_MSL512_grid_train/model_history_1.csv
```

**iii.** In order to construct a plot of the model's prediction on the UNSC corpus, simply execute the following with the path to the prediction `csv` file, which should have been automatically generated in the prediction step:

```shell
$ Rscript vis.R --predictions ./data/UNSC/pred/pred_tokens_stats_512.csv
```

### 5. Acknowledgments

**@kpe** for BERT/ALBERT code in [bert-for-tf2](https://github.com/kpe/bert-for-tf2)
