import sys, trace, re
import torch
from util import load_jsonl, dump_jsonl
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from sort import *

f = open("sort.py")
lines = f.read().splitlines()

prev_vars = {}
prev_changed_var = ''
trace = ''

def insert_number_spaces(s):
    space_before = re.sub(r"(\d)", r" \1", s)
    space_after = re.sub(r"(\d)(\S)", r"\1 \2", space_before)
    return space_after

def simple_template(l):
    list_l = list(l)
    sort_l = sorted(l)
    return {"prompt": f"sort({list_l})=", "response": "ANSWER: " + sort_l}

def custom_trace(frame, event, arg = None):
  global prev_vars, prev_changed_var, trace
  #print(event, frame.f_lineno, frame.f_code, frame.f_locals)
  line_no = frame.f_lineno
  code_line = lines[line_no - 1].strip()
  local_vars = frame.f_locals
  #print(prev_vars, local_vars)
  relevant_vars = {k:v for (k,v) in local_vars.items() if k not in prev_vars or not prev_vars[k] == local_vars[k] or k == prev_changed_var}
  #print(relevant_vars)
  prev_changed_var = code_line.split("=")[0].strip()
  prev_vars = local_vars.copy()
  if len(relevant_vars) > 0:
    trace += ", ".join([str(k) + " = " + str(v) for (k, v) in relevant_vars.items()]) + '\n'
  trace += code_line + '\n'
  return custom_trace

def chain_of_thought_template(l):
    global trace

    l = list(l)
    sort_l = sorted(l)
    str_ans = sort_l

    sys.settrace(custom_trace)
    #ret = addition('123', '1234')
    ret = sort_integers(l)
    sys.settrace(None)

    prompt = f"sort({l})="
    response = trace + f"ANSWER: {str_ans}"

    trace = ''

    prompt = insert_number_spaces(prompt)
    response = insert_number_spaces(response)

    return {"prompt": prompt, "response": response}

# Generate dataset summing up to ten digit numbers
def gen_dataset(prompt_template, num_samples=100000, max_num=int(1e5), min_len = 2, max_len=10):
    # Create dataset
    lens = np.random.randint(min_len, high = max_len + 1, size = num_samples)
    lists = [np.random.randint(0, max_num, size = l) for l in lens]
    dataset = []
    for i, l in tqdm(enumerate(lists)):
        dataset.append(prompt_template(l))
    return dataset


if __name__ == "__main__":
    dataset_dir = "datasets/"

    prompt_template = chain_of_thought_template
    #prompt_template = simple_template
    # First make clean dataset
    num_train_samples = 100000
    num_test_samples = 1000
    file_name = dataset_dir + "sort_{}_{}".format(prompt_template.__name__, num_train_samples)
    train_clean_dataset = gen_dataset(prompt_template=prompt_template, num_samples=num_train_samples)
    test_clean_dataset = gen_dataset(prompt_template=prompt_template, num_samples=num_test_samples)

    Path(file_name).mkdir(exist_ok=True, parents=True)
    dump_jsonl(os.path.join(file_name, "train.jsonl"), train_clean_dataset)
    dump_jsonl(os.path.join(file_name, "test.jsonl"), test_clean_dataset)

