import sys, trace, re
from typing import List
import torch
from pathlib import Path
import os, random
from tqdm import tqdm
import numpy as np
import operator
import multiprocessing as mp
import random
from functools import partial
import argparse
import yaml
import json

from util import dump_jsonl
from func import *


######## TInt trace utility ########

codefile = "func.py"
f = open(codefile)
lines = f.read().splitlines()

prev_vars = {}
prev_changed_var = ''
trace = ''
trace_line_limit = 300
exec_depth = 0
subbed_line = ""
include_code = False

class TraceLimitExceeded(Exception):
    pass


def reset_vars():
    global prev_vars, prev_changed_var, trace, exec_depth, subbed_line
    prev_vars = {}
    prev_changed_var = ''
    trace = ''
    exec_depth = 0
    subbed_line = ""


def insert_number_spaces(s):
    space_after = re.sub(r"(\d)([^\s])", r"\1 \2", s)
    # hack - do it twice for overlapping patterns
    space_after = re.sub(r"(\d)([^\s])", r"\1 \2", space_after)
    space_before = re.sub(r"([^\s])(\d)", r"\1 \2", space_after)
    return space_before


def replace_bases(t):
    t = re.sub(r"([a-zA-Z0-9_]+).__add\(([a-zA-Z0-9_]+)\)", r"\1 + \2", t)
    t = re.sub(r"([a-zA-Z0-9_]+).__sub\(([a-zA-Z0-9_]+)\)", r"\1 - \2", t)
    t = re.sub(r"([a-zA-Z0-9_]+).__mul\(([a-zA-Z0-9_]+)\)", r"\1 * \2", t)
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
  global codefile, prev_vars, prev_changed_var, trace, exec_depth, subbed_line, include_code, trace_line_limit
  line_no = frame.f_lineno
  if codefile != "func.py": print("CODEFILE CHANGED: ", codefile + "\n")
  if not codefile in frame.f_code.co_filename: 
    return custom_trace
  code_line = lines[line_no - 1].strip()
  local_vars = frame.f_locals
  if local_vars["vis"] == INVIS:
    subbed_line = ""
    return custom_trace
  elif local_vars["vis"] == VIS:
    if event == "call":
      if len(subbed_line) > 0: trace += subbed_line + "\n"
    subbed_line = ""
    pass # keep trace
  elif local_vars["vis"] == CALL:
    #if exec_depth > 2: return custom_trace
    if event == "call":
      # Only keep trace for top level call of function
      if len(subbed_line) > 0: trace += subbed_line + "\n"
    subbed_line = ""
    return custom_trace
  else:
    raise ValueError("Unknown visibility {}".format(local_vars["vis"]))

  if include_code and event not in ["call"] and "return" not in code_line:
    trace += code_line + "\n"

  relevant_vars = {k:v for (k,v) in local_vars.items() if k not in prev_vars or not prev_vars[k] == local_vars[k] or k == prev_changed_var}
  if len(relevant_vars) > 0:
    formatted_vars = [str(k) + " = " + str(v) for (k, v) in relevant_vars.items() if k != "vis"]
    if len(prev_vars) == 0 or len(formatted_vars) >= 1:
      trace += ", ".join(formatted_vars) + '\n'

  prev_changed_var = code_line.split("=")[0].strip()
  prev_vars = local_vars.copy()
  # add some code line with rhs substituted in
  sides = code_line.split("=")
  if len(sides) != 2: 
    subbed_line = ""
    return custom_trace
  [code_lhs, code_rhs] = sides
  vars_list_by_length = list(local_vars.keys())
  vars_list_by_length.sort(key=len, reverse=True)
  for var in vars_list_by_length:
    code_rhs = code_rhs.replace(var, str(local_vars[var]))
  subbed_line = code_lhs + "= call(" + code_rhs + ")"

  # If len of trace exceeds trace_line_limit then terminate execution
  if len(trace.split("\n")) - 1 > trace_line_limit: raise TraceLimitExceeded

  return custom_trace


######## Templates ########

def no_template(op_string, 
                noise_fn,
                line_limit,
                *args):
    ops = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.floordiv,
            "len": lambda x: x.len(),
            "rsh": operator.rshift,
            "rind": operator.getitem,
            "sort": sort_ints,
            "gcd": euclidean_alg,
            "median": median,
        }
    op_to_prompt = {
                    "add": "{} + {}",
                    "sub": "{} - {}",
                    "mul": "{} * {}",
                    "div": "{} / {}",
                    "len": "len({})",
                    "rsh": "{} >> {}",
                    "rind": "{}[{}]",
                    "sort": "sort({})",
                    "gcd": "gcd({}, {})",
                    "median": "median({})",
                }

    # switch args if x < y
    if op_string == "sub":
        if args[0] < args[1]:
            args = args[::-1]
    if op_string == "div":
        if args[0] < args[1] and args[0] > TInt(0):
            args = args[::-1]
    if op_string == "sort":
        args = [list(args)]
    if op_string == "median":
        args = [list(args)]

    ret = ops[op_string](*args)
    ret = TInt(int(ret))
    ret = noise_fn(ret)

    prompt = op_to_prompt[op_string].format(*args) + "\n"
    response = f"{ret}"

    prompt = insert_number_spaces(prompt)
    response = insert_number_spaces(response)

    return {"prompt": prompt, "response": response}


def chain_of_thought_template(op_string, 
                              noise_fn,
                              *args):
    global trace, trace_line_limit

    ops = {"add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.floordiv,
            "len": lambda x: x.len(),
            "rsh": operator.rshift,
            "rind": operator.getitem,
            "sort": sort_ints,
            "gcd": euclidean_alg,
            "median": median,
    }
    op_to_prompt = {
                    "add": "{} + {}",
                    "sub": "{} - {}",
                    "mul": "{} * {}",
                    "div": "{} / {}",
                    "len": "len({})",
                    "rsh": "{} >> {}",
                    "rind": "{}[{}]",
                    "sort": "sort({})",
                    "gcd": "gcd({} , {})",
                    "median": "median({})"
                }

    # switch args if x < y
    if op_string == "sub":
        if args[0] < args[1]:
            args = args[::-1]
    if op_string == "div":
        if args[0] < args[1] and args[0] > TInt(0):
            args = args[::-1]
    if op_string in ["sort", "median"]:
        args = [list(args)]

    sys.settrace(custom_trace)
    try:
        ret = ops[op_string](*args)
    except TraceLimitExceeded:
        ret = O
    sys.settrace(None)
    ret = TInt(int(ret))

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

    # Prepend variable namings to prompt
    prompt = op_to_prompt[op_string].format(*args) + "\n"
    response = trace + f"{ret}"

    prompt = insert_number_spaces(prompt)
    # Statically noise the response
    response = noise_fn(response)
    response = insert_number_spaces(response)

    # Reset global trace and other vars
    reset_vars()

    return {"prompt": prompt, "response": response}


def get_template(template_name):
    if template_name == "no_template":
        return no_template
    elif template_name == "chain_of_thought":
        return chain_of_thought_template
    else:
        raise ValueError(f"Unknown template: {template_name}!!!")


######## Integer sampling utilities ########

def sample_by_len(min, 
                  max,
                  num_samples,):
    """
    Sample integer arguments by randomly sampling each digit
    + min: minimum length of argument
    + max: maximum length of argument
    + num_samples: number of arguments to generate
    """
    lens = torch.randint(min, max + 1, (num_samples,)).tolist()
    nums = []
    for l in lens:
        if l == 1:
            num = torch.randint(0, 10, (1,)).tolist()
        else:
            num = torch.randint(1, 10, (1,)).tolist() + torch.randint(0, 10, (l-1,)).tolist()
        num = int("".join([str(e) for e in num]))
        nums.append(TInt(num))
    return nums


def sample_by_magnitude(min, 
                        max,
                        num_samples,):
    """
    Sample integer arguments by sampling uniformly on a range [min_int, max_int]
    + min: min
    + max: max
    + num_samples: num integers to sample
    """
    nums = torch.randint(min, max, (num_samples,))
    return [TInt(num.item()) for num in nums]


def sample_terms(min, 
                 max,
                 n_args,
                 sampling_mode,
                 num_samples,):
    """
    Sample integer arguments
    + min: minimum sampling bound
    + max: maximum sampling bound
    + n_args: number of integers to draw per sample
    + sampling_mode: how to sample integers
    + num_samples: num samples to draw
    """
    if sampling_mode == "len":
        sampler = sample_by_len
    elif sampling_mode == "mag":
        sampler = sample_by_magnitude
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}!!!")

    arg_list = [sampler(min=min, max=max, num_samples=num_samples) for _ in range(n_args)]
    arg_list = list(zip(*arg_list))
    return arg_list


######## Noise functions ########

def add_char_noise(trace, 
                   char_noise,):
    """
    Add noise to trace digit by digit
    + trace: trace to noise
    + char_noise: probability of noising each digit
    """
    s = trace.split("\n")
    t = []
    for line in s:
        l = []
        for word in line.split(" "):
            w_res = ''
            if word.isnumeric():
                for c in word:
                    p_corrupt_char = torch.rand(1)
                    if p_corrupt_char < char_noise:
                        w_res += str(torch.randint(0, 10, (1,)).item())
                    else:
                        w_res += c
            else:
                w_res = word
            l.append(w_res)
        t.append(" ".join(l))
    t = "\n".join(t)
    return t


def add_line_noise(trace, 
                   line_noise,):
    """
    Add line noise to datapoint trace
    + trace: trace to add line noise to
    + line_noise: probability of corrupting a line
    """
    core = trace.split("\n")[:-1]
    answer = trace.split("\n")[-1]
    samples = np.random.rand(len(core))
    new_core = []
    for i, line in enumerate(core):
        if samples[i] > line_noise:
            new_core.append(line)
    trace = "\n".join(new_core) + "\n" + answer
    return trace


def add_static_noise(trace, 
                     doc_noise, 
                     line_noise, 
                     char_noise,):
    """
    Add static noise to trace
    + trace: trace to add static noise to
    + doc_noise: probability of corrupting trace
    + line_noise: probability of corrupting a line
    + char_nosie: probability of corrupting a digit
    """
    doc_sample = torch.rand((1,)).item()
    if doc_sample < doc_noise:
        trace = add_char_noise(trace, char_noise)
        trace = add_line_noise(trace, line_noise)
    return trace


####### Core data generation pipeline ########

def _gen_noisy_dataset(op_name,
                       prompt_template,
                       noise_fn,
                       num_samples,
                       sampler_fn,
                       rank, 
                       save_dict,):
    print(f"Sampling {op_name}...")
    # Sample integer arguments
    samples = sampler_fn(num_samples=num_samples)
    data = [prompt_template(op_name, noise_fn, *sample) for sample in samples]
    save_dict[rank] = data
    return data


def gen_noisy_dataset(save_folder,
                      op_name,
                      prompt_template,
                      new_trace_line_limit,
                      visible_ops,
                      invisible_ops,
                      doc_noise,
                      line_noise,
                      char_noise,
                      dynamic_noise,
                      sampling_mode, 
                      arg_min_size,
                      arg_max_size,
                      n_args,
                      num_train,
                      num_test,
                      num_procs=1,):
    torch.random.seed()
    # Set trace_line_limit
    global trace_line_limit
    trace_line_limit = new_trace_line_limit
    # Reset op visibilities 
    TInt.reset_vis()
    [TInt.update_vis(op_name, INVIS) for op_name in invisible_ops]
    [TInt.update_vis(op_name, VIS) for op_name in visible_ops]
    # Set dynamic noise level
    TInt.set_dynamic_noise(dynamic_noise)

    # Set noise_fn
    noise_fn = partial(add_static_noise,
                       doc_noise=doc_noise,
                       line_noise=line_noise,
                       char_noise=char_noise,)
    # Set argument sampler_fn
    sampler_fn = partial(sample_terms,
                         min=arg_min_size,
                         max=arg_max_size,
                         n_args=n_args,
                         sampling_mode=sampling_mode,)
    # Set prompt template
    prompt_template = get_template(prompt_template)

    # Generate train data using multiple procs
    manager = mp.Manager()
    save_dict = manager.dict()
    procs = []
    for i in range(num_procs):
        assert num_train % num_procs == 0
        num_proc_samples = num_train // num_procs
        p = mp.Process(target=_gen_noisy_dataset, args=(op_name,
                                                        prompt_template,
                                                        noise_fn,
                                                        num_proc_samples,
                                                        sampler_fn,
                                                        i, 
                                                        save_dict,))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    train_dataset = [s for l in save_dict.values() for s in l]
    # Generate test data using one proc
    test_dataset = _gen_noisy_dataset(op_name,
                                      prompt_template,
                                      noise_fn=noise_fn,
                                      num_samples=num_test,
                                      sampler_fn=sampler_fn,
                                      rank=0,
                                      save_dict={},)
    
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    save_path = os.path.join(save_folder, op_name)
    save_file = f"{prompt_template.__name__}_\
{doc_noise}_{line_noise}_{char_noise}_{dynamic_noise}_\
{sampling_mode}_{arg_min_size}_{arg_max_size}"
    save_path = os.path.join(save_path, save_file)
    print(f"Saving dataset in {save_path}...")

    Path(save_path).mkdir(exist_ok=True, parents=True)
    dump_jsonl(os.path.join(save_path, "train.jsonl"), train_dataset)
    dump_jsonl(os.path.join(save_path, "test.jsonl"), test_dataset)

    # Compute stats about generated datasets
    def compute_stats(dataset):
        stats = dict()
        stats["num_samples"] = len(dataset)
        line_counts = np.array([len(sample["response"].split("\n")) - 1 for sample in dataset], dtype=np.int32)
        stats["avg_trace_line_count"] = float(np.mean(line_counts))
        stats["max_trace_line_count"] = int(np.max(line_counts))
        stats["num_at_max_trace_line_count"] = int(np.sum(line_counts >= trace_line_limit))
        stats["trace_line_limit"] = trace_line_limit
        return stats

    stats = dict()
    stats["train"] = compute_stats(train_dataset)
    stats["test"] = compute_stats(test_dataset)
    stats["config"] = dict(
                        save_folder=save_folder,
                        op_name=op_name,
                        prompt_template=prompt_template.__name__,
                        new_trace_line_limit=new_trace_line_limit,
                        visible_ops=visible_ops,
                        invisible_ops=invisible_ops,
                        doc_noise=doc_noise,
                        line_noise=line_noise,
                        char_noise=char_noise,
                        dynamic_noise=dynamic_noise,
                        sampling_mode=sampling_mode, 
                        arg_min_size=arg_min_size,
                        arg_max_size=arg_max_size,
                        n_args=n_args,
                        num_train=num_train,
                        num_test=num_test,
                        num_procs=num_procs,
                    )
    with open(os.path.join(save_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="datasets/data/")
    parser.add_argument("--config_path", type=str, default=None)
    
    parser.add_argument("--prompt_template", type=str, default="chain_of_thought", choices=["chain_of_thought", "no_template"])
    parser.add_argument("--sampling_mode", type=str, default="len", choices=["len", "mag"])
    parser.add_argument("--arg_min_size", default=1, type=int)
    parser.add_argument("--arg_max_size", default=10, type=int)
    parser.add_argument("--n_args", type=int, default=2, help="Number of arguments to sample")

    parser.add_argument("--op_name", type=str, default="add", choices=["__eq__", "__ne__", "__gt__", "__ge__",
                                                                       "__or__", "__getitem__", "__rshift__", "__lshift__",
                                                                       "__add__", "__sub__", "__mul__", "__floordiv__"])

    parser.add_argument("--invisible_ops", default=[], type=List[str], help="By default all lower level ops are invisible")
    parser.add_argument("--visible_ops", default=[], type=List[str], help="By default all higher level ops are visible")
    parser.add_argument("--new_trace_line_limit", default=300, help="Maximum number of lines in generated trace before termination")

    parser.add_argument("--doc_noise", default=0, type=float)
    parser.add_argument("--line_noise", default=0, type=float)
    parser.add_argument("--char_noise", default=0, type=float)
    parser.add_argument("--dynamic_noise", default=0, type=float)

    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=500)

    parser.add_argument("--num_procs", default=1, help="Num procs used to generate data")

    args = parser.parse_args()
    args_dict = vars(args)

    config_path = args_dict.pop("config_path")
    if config_path is not None:
       with open(config_path, "r") as f:
        config = yaml.safe_load(f)
       args_dict.update(config)

    print("Settings: ")
    print(json.dumps(args_dict, indent=2))
    gen_noisy_dataset(**args_dict)
