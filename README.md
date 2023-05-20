# transformer arithmetic

This is a repository for testing language model robustness to training data noise. 

## Datasets

Under the `data` directory is a script `data/data.py` for generating datasets in jsonl format. Datasets are named as `additions_<noise_level>_<template>.jsonl` where:

 - `<noise_level>` is the probability noise is added
 - `<template>` is a function for generating the prompt

## Training

Also included are scripts for training and testing basic models. 

### Requirements

To run, we assume the following system requirements: 

[TODO]

### Run

Then the following script may be run:

```
./scripts/train.sh
```

### Notes

We can distill via sequential sparsification

Removing code lines significantly improves performance and decreases size

Fine-tuning on addition then subtraction might improve subtraction performance.

Plus and minus overwrite each other if not training on mixed dataset