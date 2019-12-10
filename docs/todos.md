## Developments

### Argumentation workflow

#### Architecture
* TODO explore different tasks -> single/double candidate and span idenitfication (most important), or three/four-way joint model
* TODO try novel architectures for seq2seq task, egs. GRU, transformer, BERT pre-trained models
* if working with three-way task, need to think of how to pass a gradient on non-existent examples -> perhaps some kind of negative sampling procedure

#### Sequence encoding
* TODO make datasets first with single and dual view (most important), and alternatively think of how to encode a tree for training (additional for three/four-way model)
* TODO make character and word representations with both cased and non-cased views -> will amount to 2^3=8 different datasets
* need to split into smaller speech segments or paragraphs to pass into pipelime
* experiment specific entity/token masking to prevent domain-specific bias from training vocabulary
* add different classes in unknown vocabulary -> such as unknown noun, unknown adjective etc.

#### Extra steps
* ibm argumentation dataset
* coreference resolution for tree structures
* try genereous claims and premises creation and map via negative sampling to actual trees and redundant candidates

### Documentation
* add all dependencies and information on how to install
* add information on init.sh and how to use
