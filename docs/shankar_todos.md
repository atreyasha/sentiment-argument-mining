Developments
------------

### Argumentation workflow

1.  Training pipeline

    1.  **TODO** cuda comptability pipeline, contact
        maintainer

    2.  **TODO** add maximum sequence option and data stats
        printing before pushing on with training

    3.  **TODO** add grid-search to workflow and add logging
        to csvs/folders with saved models

    4.  **TODO** add appropriate citations for code, review
        to make sure this is done correctly

    5.  add grid-search json to help with choices defined on disk

    6.  possible script for continue training if patience not triggered

    7.  when converting to graph, mask out N to zero in adjacency matrix

    8.  look into argument structure and ensure all arguments are
        present in same paragraph

2.  Sequence encoding

    1.  **TODO** improve multitask data processing pipeline
        with task specification and complete json corpus with argument
        structure as matrix

    2.  **TODO** improve splits in next runs with more
        thought put behind into distribution of splits

    3.  find shorter sequence candidates in UNSC corpus for testing out
        model

3.  Architecture

    1.  **TODO** if there are still OOM issues -\> look at
        **run.ai** for accumulation optimzers and implement training
        generators

    2.  **TODO** work on task 1 and observe how multi-task
        setting could improve both tasks, use **adjacency matrix** for
        second task

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
