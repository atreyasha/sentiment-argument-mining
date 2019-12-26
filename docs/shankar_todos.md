Developments
------------

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

### Argumentation workflow

1.  Sequence encoding

    1.  **TODO** make character and (simple) word
        representations with both cased and non-cased views -\> OOM
        issues for character-transformer model

    2.  **TODO** save numpy binaries for continued use

    3.  **TODO** ultimately extend encodings to tree
        structures with distances to connective relation

    4.  need to split into smaller speech segments or paragraphs to pass
        into pipeline

    5.  1: claim, 2: claim connecting premise, 3: premise connecting
        premise (behind), 4: no tag, 5: void

2.  Architecture

    1.  **TODO** perform single task first, and then multi
        task to check performance

    2.  try novel architectures for seq2seq task, egs. GRU, transformer,
        BERT pre-trained models

    3.  think of best unique tree structure classification, perhaps with
        argument connection distances

    4.  if working with three-way task, need to think of how to pass a
        gradient on non-existent examples -\> perhaps some kind of
        negative sampling procedure

3.  Domain debiasing

    1.  **TODO** remove capital names and references to
        reduce bias

    2.  experiment specific entity/token masking to prevent
        domain-specific bias from training vocabulary

    3.  perhaps collapse all first, second and third-person pronouns to
        prevent self-referential bias

    4.  add different classes in unknown vocabulary -\> such as unknown
        noun, unknown adjective etc.

4.  Ideas to extrapolate

    1.  ibm argumentation dataset

    2.  coreference resolution for tree structures

    3.  try genereous claims and premises creation and map via negative
        sampling to actual trees and redundant candidates

5.  Documentation

    1.  **TODO** fill up pydocstrings for publishable
        functions

    2.  add all dependencies and information on how to install

    3.  add information on init.sh and how to use
