## Sentiment and Argumentation Mining (UN Security Council Speeches)

### Table of Contents

* [Overview](#Overview)
* [Methodologies](#Methodologies)
* [Citations](#Citations)
* [Developments](#Developments)
* [Authors](#Authors)

### Overview

This project entails sentiment and argumentation mining into the recently published UN security council speeches (detailed in [Schönfeld et al. 2019](#Citations)), which is publicly accessible under the link below:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH

This dataset contains ~65,000 UN security council speeches from 4,400 security council meetings from years 1995-2017. Each meeting is split up into the various speeches given by member countries. Furthermore, speeches are annotated with dates, topics and overall meeting outcomes.

It could be very interesting to conduct sentiment and argumentation analysis on this dataset, as it is generally uncommon to find textual corpora for political text and debates.

### Methodologies

#### Sentiment Mining

Here, we envision conducting sentiment mining and emotion recogniton techniques to identify the polarity of various speeches and/or speech segments.

A detailed list of ideas/methodologies can be found [here](/docs/sentiment.md)

#### Argumentation Mining

Here, we envision conducting argumentation mining to break the UNSC corpus down into claims and premises; which could be used in downstream tasks such as classification of spontaneous and/or prepared speech (segments). 

A detailed list of ideas/methodologies can be found [here](/docs/arguments.md)

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

A detailed development log can be found [here](/docs/todos.md).

### Authors

Atreya Shankar, Juliane Hanel

Project Module: Mining Sentiments and Arguments, WiSe 2019/20

Cognitive Systems: Language, Learning, and Reasoning, University of Potsdam
