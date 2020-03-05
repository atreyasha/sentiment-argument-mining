## Sentiment and Argumentation Mining (UN Security Council Speeches)

This readme will summarize our code and results in conducting sentiment analysis and argumentation mining on the UNSC corpus.

### Table of Contents

1. [Dependencies](#1-Dependencies)
2. [Repository initialization](#2-Repository-initialization)

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
> install.packages(c("ggplot2","tikzDevice","reshape2","optparse"))
```
### 2. Repository initialization

In order to initialize this repository, simply run `init.sh` as shown below:

```shell
$ ./init.sh
```

1. Firstly, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

2. Secondly, you will be prompted to download and deploy the UNSC and US election debate corpus. This will download and unzip the corresponding files.
