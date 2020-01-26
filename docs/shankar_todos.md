Developments
------------

1. Fix up metrics class with epoch-wise f1 checks for callbacks -\> use val loss for early stopping
===================================================================================================

2. Fix up logging for models and reconstruction information, and accuracies (perhaps with tensorboard)
======================================================================================================

3. Fix up grid search with various parameters and model end types
=================================================================

4. Test this out, compute computation time for 1 week and try out various optimizations for performance
=======================================================================================================

### Argumentation workflow

1.  Training pipeline

    1.  **TODO** make model more agnostic to exact label
        numbers etc.

    2.  **TODO** change printing of learning rate scheduler
        to be prettier

    3.  **TODO** make specific callback class for f1 metric
        instead of metrics to avoid batch approximation error

    4.  **TODO** consider how to **mask** outputs to get only
        relevant data and pre-assign others, and proceed with loss over
        those

    5.  **TODO** add appropriate citations for code, review
        to make sure this is done correctly

    6.  **TODO** think about extra labels for \"X\" class,
        also **custom accuracy metrics** for particular classes instead
        of averaged, perhaps also f1

    7.  **TODO** add maximum sequence option and data stats
        printing before pushing on with training

    8.  **TODO** add grid-search to workflow and add logging
        to csvs/folders with saved models -\> find tensorflow-specific
        way of doing this

    9.  when converting to graph, mask out N to zero in adjacency matrix

    10. look into argument structure and ensure all arguments are
        present in same paragraph

2.  Sequence encoding

    1.  **TODO** place training/test data in convenient
        locations for caching and re-using, perhaps move conversion to
        id\'s and saving into pre~process~ directly

    2.  **TODO** improve splits in next runs with more
        thought put behind into distribution of splits

    3.  **TODO** fix up data structure with different tasks
        later on, perhaps can merge all tasks into one, or keep multiple
        tasks, eg. single json for corpus etc.

    4.  find shorter sequence candidates in UNSC corpus for testing out
        model

3.  Architecture

    1.  **TODO** attempt using tensorboard for better
        visualization and understanding

    2.  **TODO** if there are still OOM issues, collect
        samples and gradients and update later -\> look at **run.ai**
        for optimizers with this functionality

    3.  **TODO** work on task 1 and observe how multi-task
        setting could improve both tasks, use **adjacency matrix** for
        second task

    4.  **TODO** think of appropriate performance metrics
        given label/tag imbalance

    5.  add various parameters such as window size for errors,
        perplexity, accuracy, bleu score for diversity

    6.  add checkpoints and early stoppage to find better models in
        training

4.  Domain debiasing

    1.  **TODO** remove capital names and references to
        reduce bias

    2.  **TODO** re-sampling or gradient weighting to
        re-train inputs with rare words more than common words

    3.  perhaps collapse all first, second and third-person pronouns to
        prevent self-referential bias

5.  Code-specifc development

    1.  **TODO** update all readmes and pydocstrings, check
        unused imports and code health in general

    2.  add existing folder checks, creation if missing and trailing
        slash addition

    3.  figure out pip local environment and how to fix this for future
        development

    4.  find out how to include fixed names into requirements.txt file
        such as tensorflow, despite no explicit call in script

    5.  add log files and model folders like other ML projects, where
        detailed reconstruction information for models can be stored
        along with many performance metrics and example runs

6.  Task construction

    1.  task 1 -\> 1: claim, 2: premise, 3: non-argument

    2.  task 2 (dependent on task 1) -\> form argumentation structure
        with adjacency matrix, multiply input from task 1 by row

7.  Story for presentation

    1.  clause extraction did not show reliable results with benepar and
        hard to process

    2.  mention using various \[SEP\] indicators for flipping sentences
        (need some more backup information for this process)

    3.  mention memory issues related to bert, therefore trying albert
        with single gpu -\> talk about differences between albert and
        bert

    4.  also shorter sequence length due to memory issues, makes for
        better toy examples

8.  Ideas to explore

    1.  ibm argumentation dataset

    2.  coreference resolution for tree structures

    3.  try genereous claims and premises creation and map via negative
        sampling to actual trees and redundant candidates
