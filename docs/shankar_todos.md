Developments
------------

### Argumentation workflow

1.  Documentation/Visualization

    1.  **TODO** push final paper and bibtex when all is
        done, add global references to results from paper

    2.  **TODO** possibly improve final output and format to
        all claims and premises extracted from speeches

    3.  add script to compute classification report of best model at the
        end of training

    4.  add test scripts to ensure constant sanity checks, or test
        manually on jarvis

2.  Sequence encoding

    1.  simple (task 1) -\> 1: claim, 2: premise, 3: non-argument

    2.  tree (task 2) -\> adjacency matrix connecting to various heads
        which could be claims or connecting premise

    3.  minor debiasing done with removal of capital references

3.  Architecture

    1.  limited time issue: make homogeneous train/validation/test
        datasets for appropriate evaluation

    2.  limited time issue: multi-task setting with argumentation tree

    3.  limited time issue: develop gradient accumulator/checkpointer in
        optimizer to save memory and use most of data available
