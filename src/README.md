## Sentiment and Argumentation Mining (UN Security Council Speeches)

This readme will summarize our code and results in conducting sentiment analysis and argumentation mining on the UNSC corpus.

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

a. Firstly, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

b. Secondly, you will be prompted to download and deploy the UNSC and US election debate corpus. This will download and unzip the corresponding files.

### 3. Pre-process US election debate data

In order to pre-process and encode US election debate data, we have developed some useful functions in `pre_process.py`. Usage documentation for this script is shown below: 

```
$ python3 pre_process.py --help

usage: pre_process.py [-h] [--dtype DTYPE]

optional arguments:
  -h, --help     show this help message and exit
  --dtype DTYPE  which type of data pre-processing; either 'tokens', 'char' or 'both' (default: tokens)
```

An example of running the script is shown below:

```shell
$ python3 pre_process.py --dtype both
```
