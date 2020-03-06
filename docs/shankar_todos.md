Developments
------------

### Argumentation workflow

simplify all metrics to the simplest forms
==========================================

add tasks to training options, also to output files
===================================================

make appropriate grid-search paramters
======================================

simplify code and architecture to palatable scale before running grid-search again
==================================================================================

reduce learning rates in LR scheduler given small batch size, change batch size and max seq length
==================================================================================================

add all necessary documentation to code, remove redundant code
==============================================================

Architecture

1.  **TODO** add all relevant metrics into evaluation step so
    they can be re-used

2.  **TODO** use smaller or narrower search space in
    grid-search, only use val~loss~ to simplify code

3.  **TODO** first develop full baseline for task 1, then
    proceed to task 2

4.  develop gradient accumulator in optimizer to save memory and use
    most of data available

5.  make homogeneous train/validation/test datasets for appropriate
    evaluation

Domain debiasing

1.  **TODO** remove capital names and references to reduce
    bias

2.  perhaps dropout would assist in training

Sequence encoding

1.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

2.  tree (task 2) -\> adjacency matrix connecting to various heads which
    could be claims or connecting premise

3.  need to split UNSC smaller speech segments or paragraphs to pass
    into pipeline

Documentation

1.  fix up all readmes for clarity

2.  fill up pydocstrings in appropriate style for all functions

3.  add all dependencies and information on how to install, test venv on
    cluster

4.  add information on init.sh and how to use
