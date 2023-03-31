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

