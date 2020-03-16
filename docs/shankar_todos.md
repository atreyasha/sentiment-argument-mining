Developments
------------

### Argumentation workflow

1.  Documentation/Visualization

    1.  **TODO** complete corpus visualization script

    2.  **TODO** fill up pydocstrings in appropriate style
        for all functions along with readmes

    3.  **TODO** squeeze in sentiment methodologies into
        repository when done, install sentiment dependencies onto venv
        for global tracking, perhaps add sentiment analysis on jupyter
        notebook with shared dependencies

    4.  update all information to indicate procedures and articles
        involved, so users can find relevant information

    5.  include overview section in ./src to summarize all sentiment and
        argumentation techniques

    6.  possibily improve some variable naming conventions such as
        direct, and also use os.path.join for directory concatenation

    7.  push final paper and bibtex when all is done, add global
        references to results from paper

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

4.  Paper

    1.  motivate with all methods tried

    2.  mention domain debiasing with removal of references

    3.  compare with scores in paper and talk about how paper made much
        simplifications

    4.  think about all other optimizations worth mentioning, talk about
        problem with size 128 reduction

    5.  concede to using low batch size to allow for more data, add
        comparison with majority class classifier as baseline

    6.  mention all limitations and recommendations for things to do;
        along with using pytorch instead of tensorflow
