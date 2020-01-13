Developments
------------

split into various categories
=============================

one option is using FLAIR models, with lstm\'s
==============================================

another is a transformer with bert
==================================

possible to re-interpret results in accuracy form
=================================================

perhaps save data in tsv format, or similar to CONLL style format
=================================================================

consider whether pos-tagging is further necessary
=================================================

look through code and identify all areas which need changing to us election data
================================================================================

modify those parts accordingly, or alternatively transform own data to conll format
===================================================================================

add auto-creation of folders if missing
=======================================

fix slash error possibilities in path argument
==============================================

### Argumentation workflow

1.  Architecture

    1.  **TODO** start off with bert seq2seq tagger in
        tensorflow, and advance application from there

    2.  **TODO** make easy data split for temporary model
        run, weakest point is the model

    3.  **TODO** add checkpoints and early stoppage to find
        better models in training, find ways to integrate bert into
        training procedure

    4.  split data into various sets and think of useful means of
        evaluating results, add various parameters such as window size
        for errors, perplexity and other useful parameters

    5.  perform single task first, and then multi task to check
        performance

    6.  consider non-transformer approach for character data due to GPU
        OOM issues -\> perhaps adding more features to unknown words

    7.  try novel architectures for seq2seq task, egs. GRU, transformer,
        BERT pre-trained models

    8.  think of best unique tree structure classification, perhaps with
        argument connection distances

    9.  if working with three-way task, need to think of how to pass a
        gradient on non-existent examples -\> perhaps some kind of
        negative sampling procedure

2.  Sequence encoding

    1.  **TODO** redefine padding length based on UNSC
        dataset paragraph or processing lengths

    2.  **TODO** add data options with both cased and
        non-cased views

    3.  need to split UNSC smaller speech segments or paragraphs to pass
        into pipeline

    4.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

    5.  tree (task 1) -\> 1: claim, 2: aux claim connecting to same
        claim (behind), 3: premise connecting to claim, 4: aux premise
        connecting to same premise (behind), 5: non-argument

    6.  tree (task 2) -\> distances to connective argument components
        which can help form tree

3.  Domain debiasing

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

4.  Local development

    1.  **TODO** figure out pip local environment for earlier
        tensorflow version

    2.  **TODO** find out how to include fixed names into
        requirements.txt file such as tensorflow, despite no explicit
        call in script

    3.  add log files and model folders like other ML projects, where
        detailed reconstruction information for models can be stored
        along with many performance metrics and example runs

5.  Ideas to extrapolate

    1.  OOM issues for character-transformer model

    2.  ibm argumentation dataset

    3.  coreference resolution for tree structures

    4.  try genereous claims and premises creation and map via negative
        sampling to actual trees and redundant candidates

6.  Documentation

    1.  redo colab notebook to clone and reset from master branch when
        publishing

    2.  fill up pydocstrings for publishable functions

    3.  add all dependencies and information on how to install

    4.  add information on init.sh and how to use
