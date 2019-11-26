## Argumentation Mining

### Mining claims/premises

* Run an argumentation pipeline in UNSC speeches to determine claims and premises
* This could also assist in deciphering between spontaneous vs. prepared segments; but this would be a downstream task

#### Possible techniques

##### 1. Joint pointer neural network

* We could use existing joint pointer architecture developed in [Potash et al. 2017](https://arxiv.org/abs/1612.08994)
* Source code available from previous student [project](https://github.com/oguzserbetci/argmin2017) in WiSe 2017/18 

* We could train the joint pointer neural architecture on the [US election debate corpus](https://github.com/ElecDeb60To16/Dataset)
* Paper published on the corpus and baseline model performance in [Haddadan et al. 2019](https://www.aclweb.org/anthology/P19-1463/)
* This corpus is already annotated for premises and claims; we could evaluate the joint pointer architecture and compare against the paper
* If results are better than the paper, we could extend the neural architecture to the UNSC corpus
* UNSC: Requires some pre-processing using discourse connectives to identify candidate components

##### 2. Developing political-domain ontology

* Joint neural network solution would only provide an approximate solution to this problem, particulary due to lack of a global ontology for the political domain
* A lower ontology designed for the political domain would be very helpful in setting up a proper knowledge base which could be much more scalable and interpretable than machine learning solutions
* However, this would be very costly and generally difficult to do
* Some papers have attempted making ontologies, for example in the medical/cancer domain [Groza and Maria 2016](https://www.researchgate.net/publication/309917353_Mining_arguments_from_cancer_documents_using_Natural_Language_Processing_and_ontologies).
* Could be assited with some combinatorical solution to find probable argumentation trees which satisfy all rules of the ontology
