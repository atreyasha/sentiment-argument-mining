Developments
------------

### Argumentation workflow

1.  Sequence encoding

    1.  **TODO** add data options with both cased and
        non-cased views

    2.  **TODO** redefine padding length based on UNSC
        dataset paragraph or processing lengths

    3.  **TODO** consider allowing for vocabulary pruning and
        flexible padding

    4.  need to split UNSC smaller speech segments or paragraphs to pass
        into pipeline

    5.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

    6.  tree (task 1) -\> 1: claim, 2: aux claim connecting to same
        claim (behind), 3: premise connecting to claim, 4: aux premise
        connecting to same premise (behind), 5: non-argument

    7.  tree (task 2) -\> distances to connective argument components
        which can help form tree

2.  Domain debiasing

    1.  **TODO** remove capital names and references to
        reduce bias

    2.  **TODO** consider using special word embeddings and
        keep unmodified to retain word relationships

    3.  **TODO** re-sampling procedure to re-train inputs
        with rare words more than common words

    4.  **TODO** possibly add unknown token types eg.
        pos-tags, ner taggers, verb types, etc.

    5.  experiment specific entity/token masking to prevent
        domain-specific bias from training vocabulary

    6.  perhaps collapse all first, second and third-person pronouns to
        prevent self-referential bias

    7.  add different classes in unknown vocabulary -\> such as unknown
        noun, unknown adjective etc.

3.  Architecture

    1.  **TODO** perform single task first, and then multi
        task to check performance

    2.  consider non-transformer approach for character data due to GPU
        OOM issues -\> perhaps adding more features to unknown words

    3.  try novel architectures for seq2seq task, egs. GRU, transformer,
        BERT pre-trained models

    4.  think of best unique tree structure classification, perhaps with
        argument connection distances

    5.  if working with three-way task, need to think of how to pass a
        gradient on non-existent examples -\> perhaps some kind of
        negative sampling procedure

4.  Ideas to extrapolate

    1.  OOM issues for character-transformer model

    2.  ibm argumentation dataset

    3.  coreference resolution for tree structures

    4.  try genereous claims and premises creation and map via negative
        sampling to actual trees and redundant candidates

5.  Documentation

    1.  redo colab notebook to clone and reset from master branch when
        publishing

    2.  fill up pydocstrings for publishable functions

    3.  add all dependencies and information on how to install

    4.  add information on init.sh and how to use
