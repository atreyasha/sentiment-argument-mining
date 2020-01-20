Developments
------------

### Argumentation workflow

write data to json for easier handling
======================================

might run into unicode formatting error for byte symbols
========================================================

use token type ids for segment ids
==================================

improve code tidyness later on, perhaps change tokenization pipeline and add to train
=====================================================================================

add token length pruning elsewhere instead of pre-process
=========================================================

handle masking downstream as well
=================================

handle accuracy for key classes instead of paddings
===================================================

add x label and consider what kind of implementation is necessary
=================================================================

look into argument structure and ensure all arguments are present in same paragraph
===================================================================================

when converting to graph, mask out N to zero in adjaceny matrix
===============================================================

Sequence encoding

1.  **TODO** split by lengths of up to 500 and think of how
    to handle this problem, maybe add possibility of inter-sequence
    connections in graph structure

2.  **TODO** pad sequences and attempt masking activations
    for padded positions

3.  **TODO** make better splits in next runs with more
    thought put behind into distribution of splits

4.  **TODO** think of creative way to handle sequence
    shortening for UNSC dataset

5.  UNSC: need to split UNSC smaller speech segments or paragraphs to
    pass into pipeline

6.  fix up data structure with different tasks later on, perhaps can
    merge all tasks into one, or keep multiple tasks

7.  add data options with both cased and non-cased views

Architecture

1.  **TODO** attempt using tensorboard for better
    visualization and understanding

2.  **TODO** if there are still OOM issues, collect samples
    and gradients and update later

3.  **TODO** investigate sota sequence tagging and graph
    connecting networks, use recent word embedding frameworks where
    possible

4.  **TODO** work on task 1 and observe how multi-task
    setting could improve both tasks, use adjacency matrix for second
    task

5.  **TODO** think of appropriate performance metrics given
    label/tag imbalance

6.  **TODO** update documentation and pydocstrings with new
    code

7.  identify maximum sequence length (words): pad up to 1900, not
    possible for bert models

8.  make naive split into train/test/sequence: use sklearn with
    random~seed~=42

9.  add various parameters such as window size for errors, perplexity,
    accuracy, bleu score for diversity

10. add checkpoints and early stoppage to find better models in training

11. consider non-transformer approach for character data due to GPU OOM
    issues -\> perhaps adding more features to unknown words

Code-specifc development

1.  **TODO** update all readmes, check unused imports and
    code health in general

2.  **TODO** add existing folder checks, creation if missing
    and trailing slash addition

3.  **TODO** figure out pip local environment for earlier
    tensorflow version

4.  **TODO** find out how to include fixed names into
    requirements.txt file such as tensorflow, despite no explicit call
    in script

5.  fix slash error possibilities in path argument

6.  check if directory exists to prevent later error, if not make
    directory

7.  add log files and model folders like other ML projects, where
    detailed reconstruction information for models can be stored along
    with many performance metrics and example runs

Task construction

1.  first priority is task 1, followed by others

2.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

3.  tree (task 1 + task 2) -\> task 1 representation + distances to
    connective argument components which can help form tree

4.  tree (task 1 + task 2) -\> 1: claim, 2: aux claim connecting to same
    claim (behind), 3: premise connecting to claim, 4: aux premise
    connecting to same premise (behind), 5: non-argument

5.  think of best unique tree structure classification, perhaps with
    argument connection distances -\> maybe make it a sorting issue
    where vector of arguments is re-sorted

6.  if working with three-way task, need to think of how to pass a
    gradient on non-existent examples -\> perhaps some kind of negative
    sampling procedure

Domain debiasing

1.  re-sampling or gradient weighting to re-train inputs with rare words
    more than common words

2.  perhaps collapse all first, second and third-person pronouns to
    prevent self-referential bias

3.  non-BERT: remove capital names and references to reduce bias

4.  non-BERT: consider using special word embeddings and keep unmodified
    to retain word relationships

5.  non-BERT: possibly add unknown token types eg. pos-tags, ner
    taggers, verb types, etc.

6.  non-BERT: experiment specific entity/token masking to prevent
    domain-specific bias from training vocabulary

7.  non-BERT: add different classes in unknown vocabulary -\> such as
    unknown noun, unknown adjective etc.

Timeline

1.  start writing paper in end February, submit by end of March

2.  write combined paper, clarify on number of pages

Documentation

1.  fill up pydocstrings for publishable functions

2.  redo colab notebook to clone and reset from master branch when
    publishing

Ideas to explore

1.  OOM issues for character-transformer model

2.  ibm argumentation dataset

3.  coreference resolution for tree structures

4.  try genereous claims and premises creation and map via negative
    sampling to actual trees and redundant candidates
