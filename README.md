# Sentiment Analysis and Argumentation Mining in UNSC Speeches

## Overview

This project entails sentiment analysis and argumentation mining into the recently published UN security council speeches (UNSC) corpus which is publicly accessible [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH). The UNSC corpus contains ~65,000 UN security council speeches from ~5,000 security council meetings from years 1995-2017. Each meeting is split up into the various speeches given by member countries. Furthermore, speeches are annotated with dates, topics and overall meeting outcomes.

The UNSC corpus is, however, not annotated for argumentation structures and sentiment polarities. In this project, we attempt to produce automatic machine-driven sentiment and argumentation annotations for the UNSC corpus; which could aid future human-driven annotations.

To find out more about our methodologies, read the next parts of this readme. Additionally, a list of documents detailing our methodologies can be found below:

* [Preliminary presentation](/docs/prelim_presentation/main.pdf)
* [Progress-update presentation](/docs/progress_presentation/main.pdf)
* [Final Report](/docs/final_report/main.pdf)

## Dependencies

1.  We developed this repository using Python versions `3.7.*`. To sync python-based dependencies, we recommend creating a virtual environment and running the following command:

    ```shell
    $ pip install -r requirements.txt
    ```

2. We use `R` versions `3.6.*` and `ggplot` for pretty visualizations. Execute the following within your R console to get our R-based dependencies:

    ```r
    > install.packages(c("ggplot2","tikzDevice","reshape2","optparse","ggsci"))
    ```

## Repository Initialization

### Data and git-hooks

In order to prepare the necessary data and git hooks, simply run `init.sh` and you will receive the following prompts:

```shell
$ ./init.sh
```

1. You will be prompted to download and deploy the UNSC corpus files. This will download and unzip the corresponding files, but can take quite some time due to large file sizes.

2. You will be prompted to download and deploy the US Election Debate corpus which is publicly accesible [here](https://github.com/ElecDeb60To16/Dataset). This will download and unzip the corresponding files, and should be fairly quick.

3. **Optional:** Finally, you will be prompted to initialize a pre-commit hook which will keep python dependencies up-to-date in `requirements.txt`. This is only necessary if you are further developing this repository.

### Pre-trained argumentation model

In this repository, we provide our best performing argumentation mining model `./model_logs/2020_03_17_09_17_44_MSL512_grid_train/model_1.h5` as a Git [LFS](https://git-lfs.github.com/) entry.

1. If `git-lfs` was already installed on your system prior to the cloning of this repository, our best performing model should have also been cloned in the `./model_logs` directory.

2. If you installed `git-lfs` on your system after cloning this repository, execute `git lfs pull` in the repository to pull the best performing model. In case of syncing problems, check out this GitHub [issue](https://github.com/git-lfs/git-lfs/issues/325) for suggested workarounds.

## Sentiment Analysis

Under sentiment analysis, we tested two successful sentiment-analysis tools; specifically VADER and AFINN, on the UNSC corpus. For subjectivity analysis, we used TextBlob, a text processing framework for Python. Next, we evaluated the predicted results to check their quality.

For further details on sentiment analysis, check out our dedicated Jupyter  [notebook](./sentiment.ipynb).

Our final product for sentiment analysis is the following [json](./data/UNSC/sentiment_annotation.json) file which maps UNSC speech IDs to automatically produced sentiment and subjectivity scores.

## Argumentation Mining

Under argumentation mining, we fine-tuned the ALBERT language encoder with custom decoders on a small annotated political argumentation corpus known as the US Election Debate corpus. Next, we applied the fine-tuned argumentation classifier on the UNSC corpus to predict and extract argumentation candidates. 

For further details on argumentation mining, check out our dedicated [readme](./argumentation.md).

Our final products for argumentation mining are twofold; firstly being the fine-tuned ALBERT language [model](./model_logs/2020_03_17_09_17_44_MSL512_grid_train/model_1.h5) and secondly a human-readable [json](./data/UNSC/pred/pred_clean_512.json) file mapping UNSC speech IDs to token-level argumentation labels. For the `json` file, we were only able to conduct argumentation mining for shorter UNSC speeches.

## References

Schönfeld et al. 2019 (UNSC corpus)

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

Haddadan et al. 2019 (US Election Debate corpus)

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

## Authors

Atreya Shankar, Juliane Hanel

Project Module: Mining Sentiments and Arguments, WiSe 2019/20

Cognitive Systems: Language, Learning, and Reasoning, University of Potsdam
