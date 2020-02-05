## Sentiment and Argumentation Mining (UN Security Council Speeches)

### Overview

This project entails sentiment and argumentation mining into the recently published UN security council speeches (detailed in [Schönfeld et al. 2019](https://arxiv.org/abs/1906.10969), which is publicly accessible [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH).

This dataset contains ~65,000 UN security council speeches from ~5,000 security council meetings from years 1995-2017. Each meeting is split up into the various speeches given by member countries. Furthermore, speeches are annotated with dates, topics and overall meeting outcomes.

It could be very interesting to conduct sentiment and argumentation analysis on this dataset, as it is generally uncommon to find textual corpora for political text and debates.

### Methodologies

A list of documents outlining our methodologies can be found below:

* [Sentiment analysis brainstorming](/docs/sentiment.md)
* [Argumentation mining brainstorming](/docs/arguments.md)
* [Project proposal](/docs/project_description/main.pdf)
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

### Developments

A detailed development log can be found [here](/docs/shankar_todos.md).

### Authors

Atreya Shankar, Juliane Hanel

Project Module: Mining Sentiments and Arguments, WiSe 2019/20

Cognitive Systems: Language, Learning, and Reasoning, University of Potsdam
