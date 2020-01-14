Developments
------------

### Argumentation workflow

1.  Architecture

    1.  **TODO** identify all areas where changes would be
        necessary in bert code regarding own data

    2.  **TODO** perhaps focus only on task 1 instead of
        multi-task, or maybe that could the last priority

    3.  **TODO** possibly undo work tokenization via nltk and
        have it done in bert

    4.  **TODO** start off with bert seq2seq tagger in
        tensorflow, and advance application from there

    5.  **TODO** make easy data split for temporary model
        run, weakest point is the model

    6.  **TODO** add checkpoints and early stoppage to find
        better models in training, find ways to integrate bert into
        training procedure

    7.  possible to evaluate with accuracy metrics as well

    8.  split data into various sets and think of useful means of
        evaluating results, add various parameters such as window size
        for errors, perplexity and other useful parameters

    9.  perform single task first, and then multi task to check
        performance

    10. consider non-transformer approach for character data due to GPU
        OOM issues -\> perhaps adding more features to unknown words

    11. try novel architectures for seq2seq task, egs. GRU, transformer,
        BERT pre-trained models

    12. think of best unique tree structure classification, perhaps with
        argument connection distances

    13. if working with three-way task, need to think of how to pass a
        gradient on non-existent examples -\> perhaps some kind of
        negative sampling procedure

2.  Code-specifc development

    1.  **TODO** add existing folder checks, creation if
        missing and trailing slash addition

    2.  **TODO** figure out pip local environment for earlier
        tensorflow version

    3.  **TODO** find out how to include fixed names into
        requirements.txt file such as tensorflow, despite no explicit
        call in script

    4.  fix slash error possibilities in path argument, check if
        directory exists to prevent later error

    5.  add log files and model folders like other ML projects, where
        detailed reconstruction information for models can be stored
        along with many performance metrics and example runs

3.  Sequence encoding

    1.  **TODO** redefine padding length based on UNSC
        dataset paragraph or processing lengths

    2.  fix up data structure with different tasks later on, perhaps can
        merge all tasks into one, or keep multiple tasks

    3.  add data options with both cased and non-cased views

    4.  need to split UNSC smaller speech segments or paragraphs to pass
        into pipeline

    5.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

    6.  tree (task 1) -\> 1: claim, 2: aux claim connecting to same
        claim (behind), 3: premise connecting to claim, 4: aux premise
        connecting to same premise (behind), 5: non-argument

    7.  tree (task 2) -\> distances to connective argument components
        which can help form tree

4.  Domain debiasing

    1.  re-sampling procedure to re-train inputs with rare words more
        than common words

    2.  remove capital names and references to reduce bias

    3.  consider using special word embeddings and keep unmodified to
        retain word relationships

    4.  possibly add unknown token types eg. pos-tags, ner taggers, verb
        types, etc.

    5.  experiment specific entity/token masking to prevent
        domain-specific bias from training vocabulary

    6.  perhaps collapse all first, second and third-person pronouns to
        prevent self-referential bias

    7.  add different classes in unknown vocabulary -\> such as unknown
        noun, unknown adjective etc.

5.  Documentation

    1.  fill up pydocstrings for publishable functions

    2.  redo colab notebook to clone and reset from master branch when
        publishing

6.  Ideas to extrapolate

    1.  OOM issues for character-transformer model

    2.  ibm argumentation dataset

    3.  coreference resolution for tree structures

    4.  try genereous claims and premises creation and map via negative
        sampling to actual trees and redundant candidates
