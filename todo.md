experiments:
- train: addition 10
  test: addition 10
- train: subtraction 10
  test: subtraction 10
- train: multiplication 4 (by length, or combined lengths 7)
  test: multiplication 4
- train: division 4
  test: division 4
- train: addition 10, subtraction 10
  test: addition 10
  test: subtraction 10
- train: addition 10, subtraction 10, multiplication, division
  test: 

- train: addition
  train: multiplication

oom ... *activation checkpointing*
subroutine


stopping criteria
each model needs to be caled exactly the same number of times
multiple copies of the model - one per gpu
may need to have dummy ways of calling
