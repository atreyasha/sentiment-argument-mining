Developments
------------

### Argumentation workflow

1.  Training pipeline

    1.  **TODO** make automated script to repeat tests for
        best performing n models to get statistical variations

    2.  **TODO** edit all log csvs to incorporate all
        relevant information

    3.  **TODO** add classification report jsons to logging
        pipelines, add them manually for other cases by reconstructing
        model and working with it

    4.  **TODO** check classification reports to better
        introspect models and their functionalities

    5.  **TODO** add grid-search json to help with choices
        defined on disk

    6.  **TODO** look at **run.ai** for accumulation
        optimzers and implement training generators -\> can increase
        batch-size for grid-search

    7.  **TODO** work on task 1 and observe how multi-task
        setting could improve both tasks, use **adjacency matrix** for
        second task

    8.  **TODO** update models in logs to have 0 index for
        cnn and lstm **and** with/without class weights

    9.  try out different val metric

    10. possible script for continue training if patience not triggered;
        look up model reconstruction by adding custom objects

    11. when converting to graph, mask out N to zero in adjacency matrix

2.  Sequence encoding

    1.  **TODO** improve multitask data processing pipeline
        with task specification and complete json corpus with argument
        structure as matrix

    2.  **TODO** improve splits in next runs with more
        thought put behind into distribution of splits

    3.  look into argument structure and ensure all arguments are
        present in same paragraph

    4.  find shorter sequence candidates in UNSC corpus for testing out
        model

3.  Domain debiasing

    1.  **TODO** remove capital names and references to
        reduce bias

    2.  increase sequence length by using accumulation to allow more
        data to feed into network

4.  Code-specifc development

    1.  **TODO** find out how to include fixed names into
        requirements.txt file such as tensorflow, despite no explicit
        call in script, figure out pip local environment and how to fix
        this for future development

    2.  **TODO** update all readmes and pydocstrings, check
        unused imports and code health in general

    3.  add appropriate citations for code, review to make sure this is
        done correctly

    4.  add existing folder checks, creation if missing and trailing
        slash addition

5.  Task construction

    1.  task 1 -\> 1: claim, 2: premise, 3: non-argument

    2.  task 2 (dependent on task 1) -\> form argumentation structure
        with adjacency matrix, multiply input from task 1 by row

6.  Story for presentation

    1.  clause extraction did not show reliable results with benepar and
        hard to process

    2.  mention using various \[SEP\] indicators for flipping sentences
        (need some more backup information for this process)

    3.  mention memory issues related to bert, therefore trying albert
        with single gpu -\> talk about differences between albert and
        bert

    4.  also shorter sequence length due to memory issues, makes for
        better toy examples

7.  Ideas to explore

    1.  ibm argumentation dataset

    2.  coreference resolution for tree structures

    3.  try genereous claims and premises creation and map via negative
        sampling to actual trees and redundant candidates
