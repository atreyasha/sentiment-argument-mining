## Sentiment and Argumentation Mining (UN Security Council Speeches)

### Overview

This project entails sentiment and argumentation mining into the recently published UN security council speeches (detailed in [Schönfeld et al. 2019](https://arxiv.org/abs/1906.10969)), which is publicly accessible [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH).

This dataset contains ~65,000 UN security council speeches from ~5,000 security council meetings from years 1995-2017. Each meeting is split up into the various speeches given by member countries. Furthermore, speeches are annotated with dates, topics and overall meeting outcomes.

It could be very interesting to conduct sentiment and argumentation analysis on this dataset, as it is generally uncommon to find textual corpora for political text and debates.

### Methodologies

For sentiment analysis, we compare the performance of various successful textual sentiment classifiers on the UNSC corpus.

For argumentation mining, we train an argument candidate classifier on the US Election Debate corpus from [Haddadan et al. 2019](https://www.aclweb.org/anthology/P19-1463/), which is available to the public [here](https://github.com/ElecDeb60To16/Dataset). Next, we apply this classifier on the UNSC corpus to check if we can extract meaningful argumentation candidates.

A list of documents detailing our methodologies can be found below:

* [Preliminary presentation](/docs/prelim_presentation/main.pdf)
* [Progress-update presentation](/docs/progress_presentation/main.pdf)

A formal description of our code and results can be found in the [src](/src) directory.

### Citations

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
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1463",
    doi = "10.18653/v1/P19-1463",
    pages = "4684--4690"
}
```

### Developments

A detailed development log can be found [here](/docs/shankar_todos.md).

### Authors

Atreya Shankar, Juliane Hanel

Project Module: Mining Sentiments and Arguments, WiSe 2019/20

Cognitive Systems: Language, Learning, and Reasoning, University of Potsdam
