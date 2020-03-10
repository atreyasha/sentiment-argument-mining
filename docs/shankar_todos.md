Developments
------------

### Argumentation workflow

1.  Documentation/Visualization

    1.  **TODO** fill up pydocstrings in appropriate style
        for all functions

    2.  **TODO** check that training script only executes
        corpus2tokenids if data missing

    3.  **TODO** add workflow for plotting results/data and
        figure out optparse for R

    4.  **TODO** fix up all readmes for clarity, add section
        for final evaluation on UNSC corpus

    5.  **TODO** update all information to indicate
        procedures and articles involved, so users can find relevant
        information

    6.  make final pull request when all is complete

2.  Sequence encoding

    1.  **TODO** need to split UNSC smaller speech segments
        or paragraphs to pass into pipeline

    2.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

    3.  tree (task 2) -\> adjacency matrix connecting to various heads
        which could be claims or connecting premise

    4.  minor debiasing done with removal of capital references

3.  Architecture

    1.  limited time issue: make homogeneous train/validation/test
        datasets for appropriate evaluation

    2.  limited time issue: multi-task setting with argumentation tree

    3.  limited time issue: develop gradient accumulator/checkpointer in
        optimizer to save memory and use most of data available

4.  Paper

    1.  motivate with all methods tried

    2.  mention domain debiasing with removal of references

    3.  compare with scores in paper and talk about how paper made much
        simplifications

    4.  think about all other optimizations worth mentioning, talk about
        problem with size 128 reduction

    5.  mention all limitations and recommendations for things to do;
        along with using pytorch instead of tensorflow
