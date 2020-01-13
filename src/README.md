## Sentiment and Argumentation Mining (UN Security Council Speeches)

This readme will summarize our code and results in conducting sentiment analysis and argumentation mining on the UNSC corpus.

### Table of Contents

1. [Dependencies](#1-Dependencies)
2. [Repository initialization](#2-Repository-initialization)
3. [Pre-process data](#3-Pre-process-data)
4. [Run models](#4-Run-models)

### 1. Dependencies

To install python-based dependencies, simply run the following command (optionally within a virtual environment):

```shell
$ pip install -r requirements.txt
```

### 2. Repository initialization

In order to initialize this repository, simply run `init.sh` as shown below:

```shell
$ ./init.sh
```

i. Firstly, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

ii. Secondly, you will be prompted to download and deploy the UNSC and US election debate corpus. This will download and unzip the corresponding files.

### 3. Pre-process data

#### i. Argumentation

In order to pre-process and encode US election debate data, we have developed some useful functions in `pre_process_argumentation.py`. Usage documentation for this script is shown below: 

```
$ python3 pre_process_argumentation.py --help

usage: pre_process.py [-h] [--dtype str]

optional arguments:
  -h, --help   show this help message and exit
  --dtype str  which type of data pre-processing; either 'tokens', 'char' or 'both' (default: tokens)
```

An example of running the script is shown below:

```shell
$ python3 pre_process_argumentation.py --dtype both
```

### 4. Run models

#### i. Argumentation

To test a basic transformer model in the seq2seq argumentation task, one can use the script `train_argumentation.py`:

```
$ python3 train_argumentation.py --help

usage: train.py [-h] [--epochs int] [--batch-size int]

optional arguments:
  -h, --help        show this help message and exit
  --epochs int      maximum number of training epochs (default: 50)
  --batch-size int  batch size in stochastic gradient descent (default: 5)
```

The resulting model after training will be saved in the `models` directory.
