## Sentiment Analysis and Argumentation Mining (UN Security Council Speeches)

### Table of Contents

1. [Overview](#1-Overview)
2. [Dependencies](#2-Dependencies)
3. [Repository Initialization](#3-Repository-Initialization)
4. [Sentiment Analysis](#4-Sentiment-Analysis)
5. [Argumentation Mining](#5-Argumentation-Mining)
6. [Developments](#6-Developments)
7. [References](#7-References)
8. [Authors](#8-Authors)

### 1. Overview

This project entails sentiment analysis and argumentation mining into the recently published UN security council speeches (UNSC) corpus (detailed in [Schönfeld et al. 2019](https://arxiv.org/abs/1906.10969)), which is publicly accessible [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH). The UNSC corpus contains ~65,000 UN security council speeches from ~5,000 security council meetings from years 1995-2017. Each meeting is split up into the various speeches given by member countries. Furthermore, speeches are annotated with dates, topics and overall meeting outcomes.

The UNSC corpus is, however, not annotated for argumentation structures and sentiment polarities. In this project, we attempt to produce automatic machine-driven sentiment and argumentation annotations for the UNSC corpus; which could aid future human-driven annotations.

To find out more about our methodologies, read the next parts of this readme. Additionally, a list of documents detailing our methodologies can be found below:

* [Preliminary presentation](/docs/prelim_presentation/main.pdf)
* [Progress-update presentation](/docs/progress_presentation/main.pdf)
* [Final Report](/docs/final_report/main.pdf)

### 2. Dependencies

**i.** In order to set up this repository, we would need to satisfy local pythonic dependencies. If `poetry` is installed on your system, you can install dependencies and create a virtual environment automatically via the following command:

```shell
$ poetry install
```

Alternatively, you can install dependencies with `pip`:

```shell
$ pip install -r requirements.txt
```

**Note**: These dependencies were tested with python version `3.7.*`, but should work with newer versions. 

**ii.** In this repository, we use `R` and `ggplot` for visualization. Execute the following within your R console to get the dependencies:

```r
> install.packages(c("ggplot2","tikzDevice","reshape2","optparse","ggsci"))
```

**Note:** R-scripts were tested with R version `3.6.*`.

**iii.** If you want to use or reference the best argumentation classification model in this repository (which is stored in a `git-lfs` entry), you would need to install `git-lfs` for your system.

If you already had `git-lfs` installed before cloning the repository, the best model data would be downloaded alongside the clone.

If you installed `git-lfs` after cloning this repository, execute `git lfs pull` in order to recover the large model data, as per suggestions [here](https://github.com/git-lfs/git-lfs/issues/325).

### 3. Repository Initialization

In order to initialize this repository, simply run `init.sh` as shown below:

```shell
$ ./init.sh
```

**i.** Firstly, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

**ii.** Secondly, you will be prompted to download and deploy the UNSC corpus files. This will download and unzip the corresponding files, but can take quite some time due to large file sizes.

**iii.** Thirdly, you will be prompted to download and deploy the US election debate corpus. This will download and unzip the corresponding files, and should be fairly quick.

### 4. Sentiment Analysis

Under sentiment analysis, we tested two successful sentiment-analysis tools, specifically [VADER](https://github.com/cjhutto/vaderSentiment) and [AFINN](https://github.com/fnielsen/afinn), on the UNSC corpus. For subjectivity analysis, we used [TextBlob](https://github.com/sloria/TextBlob), a text processing framework for Python. Next, we evaluated the predicted results to check their quality.

For further details on sentiment analysis, check out our dedicated Jupyter  [notebook](./sentiment.ipynb).

Our final product for sentiment analysis is the following [json](./data/UNSC/sentiment_annotation.json) file which maps UNSC speech IDs to automatically produced sentiment and subjectivity scores.

### 5. Argumentation Mining

Under argumentation mining, we fine-tuned the [ALBERT](https://github.com/google-research/ALBERT) language encoder (with custom decoders) on a small annotated political argumentation corpus known as the US Election Debate (USED) Corpus, detailed in [Haddadan et al. 2019](https://www.aclweb.org/anthology/P19-1463/) and publicly available [here](https://github.com/ElecDeb60To16/Dataset). Next, we applied the fine-tuned argumentation classifier on the UNSC corpus to predict and extract argumentation candidates. 

For further details on argumentation mining, check out our dedicated [readme](./argumentation.md).

Our final products for argumentation mining are twofold; firstly being the fine-tuned ALBERT language [model](./model_logs/2020_03_17_09_17_44_MSL512_grid_train/model_1.h5) and secondly a human-readable [json](./data/UNSC/pred/pred_clean_512.json) file mapping UNSC speech IDs to token-level argumentation labels. For the `json` file, we were only able to conduct argumentation mining for shorter UNSC speeches.

### 6. Developments

A detailed development log can be found [here](/docs/shankar_todos.md).

### 7. References

Schönfeld et al. 2019 (paper describing creation of UN security council corpus)

```
@misc{schnfeld2019security,
    title={The UN Security Council debates 1995-2017},
    author={Mirco Schönfeld and Steffen Eckhard and Ronny Patz and Hilde van Meegdenburg},
    year={2019},
    eprint={1906.10969},
    archivePrefix={arXiv},
    primaryClass={cs.DL}
}
```

Haddadan et al. 2019 (paper describing US Election Debate corpus)

```
@inproceedings{haddadan-etal-2019-yes,
    title = "Yes, we can! Mining Arguments in 50 Years of {US} Presidential Campaign Debates",
    author = "Haddadan, Shohreh  and
      Cabrio, Elena  and
      Villata, Serena",
    booktitle = {Proceedings of the 57th Annual Meeting of the Association
    for Computational Linguistics},
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1463",
    doi = "10.18653/v1/P19-1463",
    pages = "4684--4690"
}
```

### 8. Authors

Atreya Shankar, Juliane Hanel

Project Module: Mining Sentiments and Arguments, WiSe 2019/20

Cognitive Systems: Language, Learning, and Reasoning, University of Potsdam
