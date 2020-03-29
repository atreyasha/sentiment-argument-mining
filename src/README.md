## Sentiment Analysis and Argumentation Mining (UN Security Council Speeches)

This readme will summarize our code and results in conducting sentiment analysis and argumentation mining on the UNSC corpus.

### Table of Contents

1. [Dependencies](#1-Dependencies)
2. [Repository Initialization](#2-Repository-Initialization)
3. [Sentiment Analysis](#3-Sentiment-Analysis)
4. [Argumentation Mining](#4-Argumentation-Mining)

### 1. Dependencies

**i.** In order to set up this repository, we would need to satisfy local pythonic dependencies. If `poetry` is installed on your system, you can install dependencies and create a virtual environment automatically via the following command:

```shell
$ poetry install
```

Alternatively, you can install dependencies with `pip`:

```shell
$ pip install -r requirements.txt
```

**Note**: These dependencies were tested with python versions `3.6.*` and `3.7.*`, but should work with newer versions. 

**ii.** In this repository, we use `R` and `ggplot` for visualization. Execute the following within your R console to get the dependencies:

```r
> install.packages(c("ggplot2","tikzDevice","reshape2","optparse","ggsci"))
```

**Note:** R-scripts were tested with R version `3.6.*`.

**iii.** If you want to use or reference the best argumentation classification model in this repository (which is stored in a `git-lfs` entry), you would need to install `git-lfs` for your system.

If you already had `git-lfs` installed before cloning the repository, the best model data would be downloaded alongside the clone.

If you installed `git-lfs` after cloning this repository, execute `git lfs pull` in order to recover the large model data, as per suggestions [here](https://github.com/git-lfs/git-lfs/issues/325).

### 2. Repository Initialization

In order to initialize this repository, simply run `init.sh` as shown below:

```shell
$ ./init.sh
```

**i.** Firstly, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

**ii.** Secondly, you will be prompted to download and deploy the UNSC corpus files. This will download and unzip the corresponding files, but can take quite some time due to large file sizes.

**iii.** Thirdly, you will be prompted to download and deploy the US election debate corpus. This will download and unzip the corresponding files, and should be fairly quick.

### 3. Sentiment Analysis

Under sentiment analysis, we tested two successful sentiment-analysis tools, specifically [VADER](https://github.com/cjhutto/vaderSentiment) and [TextBlob](https://github.com/sloria/TextBlob), on the UNSC corpus. Next, we evaluated the predicted results to check their quality.

For further details on sentiment analysis, check out our dedicated Jupyter notebook [here](./sentiment.ipynb).

### 4. Argumentation Mining

Under argumentation mining, we fine-tuned the [ALBERT](https://github.com/google-research/ALBERT) language encoder (with custom decoders) on a small annotated political argumentation corpus known as the US Election Debate Corpus, detailed in [Haddadan et al. 2019](https://www.aclweb.org/anthology/P19-1463/). Next, we applied the fine-tuned argumentation classifier on the UNSC corpus to predict and extract argumentation candidates. 

For further details on argumentation mining, check out our dedicated readme [here](./argumentation.md).
