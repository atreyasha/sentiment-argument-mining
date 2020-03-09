Developments
------------

### Argumentation workflow

1.  Architecture

    1.  **TODO** first develop full baseline for task 1, then
        proceed to task 2, think about whether multi-task setting is
        feasible given time

    2.  **TODO** add tasks to training options, also to
        output files

    3.  limited time issue: develop gradient accumulator/checkpointer in
        optimizer to save memory and use most of data available

    4.  limited time issue: make homogeneous train/validation/test
        datasets for appropriate evaluation

2.  Sequence encoding

    1.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

    2.  tree (task 2) -\> adjacency matrix connecting to various heads
        which could be claims or connecting premise

    3.  need to split UNSC smaller speech segments or paragraphs to pass
        into pipeline

3.  Documentation

    1.  **TODO** fix up all readmes for clarity

    2.  **TODO** fill up pydocstrings in appropriate style
        for all functions

    3.  add all dependencies and information on how to install, test
        venv on cluster

    4.  add information on init.sh and how to use
