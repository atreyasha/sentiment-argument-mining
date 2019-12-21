Developments
------------

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

### Argumentation workflow

1.  Architecture

    1.  **TODO** explore different tasks -\> single/double
        candidate and span idenitfication (most important), or
        three/four-way joint model

    2.  **TODO** try novel architectures for seq2seq task,
        egs. GRU, transformer, BERT pre-trained models

    3.  if working with three-way task, need to think of how to pass a
        gradient on non-existent examples -\> perhaps some kind of
        negative sampling procedure

2.  Sequence encoding

    1.  **TODO** make datasets first with single and dual
        view (most important), and alternatively think of how to encode
        a tree for training (additional for three/four-way model)

    2.  **TODO** make character and word representations with
        both cased and non-cased views -\> will amount to 2^3^=8
        different datasets

    3.  need to split into smaller speech segments or paragraphs to pass
        into pipeline

3.  Ideas to innovate

    1.  experiment specific entity/token masking to prevent
        domain-specific bias from training vocabulary

    2.  add different classes in unknown vocabulary -\> such as unknown
        noun, unknown adjective etc.

4.  Extra steps

    1.  ibm argumentation dataset

    2.  coreference resolution for tree structures

    3.  try genereous claims and premises creation and map via negative
        sampling to actual trees and redundant candidates

5.  Documentation

    1.  add all dependencies and information on how to install

    2.  add information on init.sh and how to use
