Developments
------------

### Argumentation workflow

1.  Architecture

    1.  **TODO** develop gradient accumulator in optimizer to
        save memory and use most of data available

    2.  **TODO** add all relevant metrics into evaluation
        step so they can be re-used

    3.  **TODO** use smaller or narrower search space in
        grid-search, only use val~loss~ to simplify code

    4.  first develop full baseline for task 1, then proceed to task 2

    5.  make homogeneous train/validation/test datasets for appropriate
        evaluation

2.  Domain debiasing

    1.  **TODO** remove capital names and references to
        reduce bias

    2.  perhaps dropout would assist in training

3.  Sequence encoding

    1.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

    2.  tree (task 2) -\> adjacency matrix connecting to various heads
        which could be claims or connecting premise

    3.  need to split UNSC smaller speech segments or paragraphs to pass
        into pipeline

4.  Documentation

    1.  fix up all readmes for clarity

    2.  fill up pydocstrings in appropriate style for all functions

    3.  add all dependencies and information on how to install, test
        venv on cluster

    4.  add information on init.sh and how to use