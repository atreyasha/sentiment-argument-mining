## Developments

* TODO fill up personal segments
* clean up remaining segments and make consistent
* TODO maybe potentially can use us election corpus -> goto function in vim seems accurate but not so for python read
* maybe this is python io stream specific issue
* might even be possible to do 4 joint tasks, including AC identification and type/role within tree

### Argumentation workflow
* TODO understand better how USElectionDebates is structured and transform into span and argument sets for training
* TODO make pull request with pre-processing code into master branch
* TODO explore different tasks -> single candidate and span idenitfication (most important), or three-way joint model
* TODO try novel architectures for seq2seq task, egs. GRU, transformer, BERT pre-trained models
* experiment specific entity masking to prevent domain-specific bias from training vocabulary
* if working with three-way task, need to think of how to pass a gradient on non-existent examples -> perhaps some kind of negative sampling procedure
 
### Goals for 101219 presentation
* basic claim/premise structure with connections
* descriptive statistics on USElectionDebates
* descriptive statistics on UNSC corpus
* description of joint pointer neural network
* limitation of JPNN: only identifies argument, claim and connections -> no argument type or adversarial relationships
* **optional:** basic results of running JPNN on us election debates
* steps forward towards UNSC data -> how to filter possible candidates on which to conduct self attention

### Documentation
* add all dependencies and information on how to install
* add information on init.sh and how to use

### Overall structure
* add roles and responsibilities
