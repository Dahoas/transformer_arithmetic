import sys, trace, re
import torch
from util import load_jsonl, dump_jsonl
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from func import *
import operator

codefile = "func.py"
f = open(codefile)
lines = f.read().splitlines()

prev_vars = {}
prev_changed_var = ''
trace = ''

def insert_number_spaces(s):
    space_after = re.sub(r"(\d)", r"\1 ", s)
    space_before = re.sub(r"([^\s])(\d)", r"\1 \2", space_after)
    return space_before

def replace_bases(t):
    t = t.replace("x.__add(y)", "x + y")
    t = t.replace("x.__mul(y)", "x * y")
    t = t.replace("x.__sub(y)", "x - y")
    t = t.replace("x.__floordiv(y)", "x / y")
    return t

def swap_func_call(t):
    lines = t.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if "def " in line:
            lines[i] = lines[i-1]
            lines[i-1] = line
        i += 1
    return "\n".join(lines)

def remove_doubled_lines(t):
    lines = t.split("\n")
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i] == lines[i+1]:
            lines.pop(i)
        i += 1
    return "\n".join(lines)

def simple_template(l):
    list_l = list(l)
    sort_l = sorted(l)
    return {"prompt": f"sort({list_l})=", "response": "ANSWER: " + sort_l}

def custom_trace(frame, event, arg = None):
  global prev_vars, prev_changed_var, trace
  #print(event, frame.f_lineno, frame.f_code, frame.f_locals)
  line_no = frame.f_lineno
  #print(frame.f_code.co_filename)
  if not codefile in frame.f_code.co_filename: 
    return custom_trace
  code_line = lines[line_no - 1].strip()
  local_vars = frame.f_locals
  if not local_vars["vis"]: 
    return custom_trace
  #print(prev_vars, local_vars)
  relevant_vars = {k:v for (k,v) in local_vars.items() if k not in prev_vars or not prev_vars[k] == local_vars[k] or k == prev_changed_var}
  #print(relevant_vars)
  prev_changed_var = code_line.split("=")[0].strip()
  prev_vars = local_vars.copy()
  if len(relevant_vars) > 0:
    trace += ", ".join([str(k) + " = " + str(v) for (k, v) in relevant_vars.items()]) + '\n'
  trace += code_line + '\n'
  return custom_trace

def chain_of_thought_template(arg1, arg2, op_string = "add"):
    global trace

    ops = {"add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.floordiv
    }

    sys.settrace(custom_trace)
    #ret = addition('123', '1234')
    ret = ops[op_string](arg1, arg2)
    sys.settrace(None)

    # Post-process trace
    ## Remove calls to copy
    trace = re.sub(r"copy\(([a-zA-Z0-9_]*)\)", "\\1", trace)
    # Replace O with 0 and I with 1 and empty assignment with 0
    trace = re.sub(r" O", r" 0", trace)
    trace = re.sub(r"\[O\]", "[0]", trace)
    trace = re.sub(r" I", " 1", trace)
    trace = re.sub(r"\[I\]", "[1]", trace)
    trace = re.sub(r"TInt\(([a-zA-Z0-9_]*)\)", r"\1", trace)
    trace = re.sub(r"([a-zA-Z_]+) = ([\n,])", r"\1 = 0\2", trace)
    # Replace hidden operation with symbolic operation
    trace = replace_bases(trace)
    # Remove doubled lines like returns and constant assignments
    trace = remove_doubled_lines(trace)
    # Removes vis var
    trace = re.sub(r", vis ?= ?[a-zA-Z]+", "", trace)
    # Switch order of function call and input vars
    trace = swap_func_call(trace)

    prompt = f"{op_string}({arg1}, {arg2})="
    response = trace + f"ANSWER: {ret}"

    prompt = insert_number_spaces(prompt)
    response = insert_number_spaces(response)

    # Reset global trace
    trace = ''

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
    x = TInt(12345)
    y = TInt(23535)
    #print(chain_of_thought_template(x, y, op_string="div")["response"])
    print(chain_of_thought_template(x, y, op_string="add")["response"])
    """dataset_dir = "datasets/"

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
    dump_jsonl(os.path.join(file_name, "test.jsonl"), test_clean_dataset)"""

