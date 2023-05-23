import sys, trace, re, time
import torch
from util import load_jsonl, dump_jsonl
from pathlib import Path
import os, random
from tqdm import tqdm
import numpy as np
from func import *
import operator
from transformers import AutoTokenizer
import multiprocessing as mp
from func import INVIS, VIS, CALL

codefile = "func.py"
f = open(codefile)
lines = f.read().splitlines()

prev_vars = {}
prev_changed_var = ''
trace = ''
exec_depth = 0
subbed_line = ""

def insert_number_spaces(s):
    space_after = re.sub(r"(\d)([^\s])", r"\1 \2", s)
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
  global codefile, prev_vars, prev_changed_var, trace, exec_depth, subbed_line
  #print(event, frame.f_lineno, frame.f_code, frame.f_locals)
  line_no = frame.f_lineno
  #print(frame.f_code.co_filename)
  if codefile != "func.py": print("CODEFILE CHANGED: ", codefile + "\n")
  if not codefile in frame.f_code.co_filename: 
    return custom_trace
  code_line = lines[line_no - 1].strip()
  #trace += "CUSTOM TRACE CALL: " + code_line + " " + event + "\n"
  local_vars = frame.f_locals
  #trace += "Code, event: " + code_line + event + "\n"
  #trace += "localvars: " + str(local_vars) + "\n"
  if local_vars["vis"] == INVIS:
    subbed_line = ""
    return custom_trace
  elif local_vars["vis"] == VIS:
    if event == "call":
      if len(subbed_line) > 0: trace += subbed_line + "\n"
    subbed_line = ""
    pass # keep trace
  elif local_vars["vis"] == CALL:
    #trace += "LOCALVIS=CALL\n"
    #if exec_depth > 2: return custom_trace
    if event == "call":
      #trace += "EVENT=CALL with subbedline: " + subbed_line + "\n"
      # Only keep trace for top level call of function
      if len(subbed_line) > 0: trace += subbed_line + "\n"
    subbed_line = ""
    return custom_trace
  else:
    raise ValueError("Unknown visibility {}".format(local_vars["vis"]))
  #print(prev_vars, local_vars)

  #trace += "CODELINE " + code_line + "\n"

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
  #trace += "SUBBED LINE: " + subbed_line + '\n'
  return custom_trace

def chain_of_thought_template(op_string, *args):
    global trace

    ops = {"add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.floordiv,
            "len": lambda x: x.len(),
            "rsh": operator.rshift,
            "rind": operator.getitem,
    }
    op_to_prompt = {
                    "add": "{} + {}",
                    "sub": "{} - {}",
                    "mul": "{} * {}",
                    "div": "{} / {}",
                    "len": "len({})",
                    "rsh": "{} >> {}",
                    "rind": "{}[{}]"
                }

    # switch args if x < y
    if op_string == "sub":
        if args[0] < args[1]:
            args = args[::-1]
    if op_string == "div":
        if args[0] < args[1] and args[0] > TInt(0):
            args = args[::-1]

    sys.settrace(custom_trace)
    ret = ops[op_string](*args)
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

    # Prepend variable namings to prompt
    #var_names = "xyzabcdefghijklmnopqrstuvwxyz"
    #for i in range(len(args)):
    #    trace += "{} = {}\n".format(var_names[i], args[i])
    prompt = op_to_prompt[op_string].format(*args) + "\n"
    response = trace + f"{ret}"

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

# returns inclusive uniform random sample of len then digit
# inclusive of max len
def sample_by_len(num_samples, min_len, max_len, min_mag = None):
    lens = torch.randint(min_len, max_len + 1, (num_samples,)).tolist()
    nums = []
    for l in lens:
        while True:
            if l == 1:
                num = torch.randint(0, 10, (1,)).tolist()
            else:
                num = torch.randint(1, 10, (1,)).tolist() + torch.randint(0, 10, (l-1,)).tolist()
            num = int("".join([str(e) for e in num]))
            if min_mag is None or num >= min_mag: break
        nums.append(TInt(num))
    return nums

def sample_by_magnitude(num_samples, min_int, max_int):
    nums = torch.randint(min_int, max_int, (num_samples,))
    sample = []
    for num in nums:
        sample.append(TInt(num.item()))
    return sample

# Generate dataset by sampling lengths then digits
def sample_terms(arg_sampling, num_samples = 100000):
    # args per sample
    args_per_sample = len(arg_sampling)
    arg_list = [[] for s in range(num_samples)]
    for arg in range(args_per_sample):
        sample_method = arg_sampling[arg]
        sample_type = sample_method[0]
        if sample_type == "len":
            len_min = sample_method[1]
            len_max = sample_method[2]
            mag_min = None
            if len(sample_method) == 4:
                mag_min = sample_method[3]
            arg_sample = sample_by_len(num_samples, len_min, len_max, min_mag = mag_min)
        elif sample_type == "mag":
            min_num = sample_method[1]
            max_num = sample_method[2]
            arg_sample = sample_by_magnitude(num_samples, min_num, max_num)
        else: 
            print("unrecognized sampling: ", sample_type)
            exit()
        # add args
        for s in range(num_samples):
            arg_list[s].append(arg_sample[s])
    return arg_list
        
def generate_op_data(op_string, prompt_template, num_samples, arg_sampling):
    print(f"Sampling {op_string}...")
    samples = sample_terms(arg_sampling, num_samples = num_samples)
    data = []
    for sample in tqdm(samples):
        #print(sample)
        datapoint = prompt_template(op_string, *sample)
        #print(datapoint)
        data.append(datapoint)
    return data

def generate_mix_data(prompt_template, sampling_dict, rank, save_dict):
    torch.random.seed()
    data = []
    for op, sampling_params in sampling_dict.items():
      TInt.reset_vis()
      for f in sampling_params["visibility"]:
        TInt.update_vis(f, sampling_params["visibility"][f])
      data += generate_op_data(op, prompt_template, sampling_params["num_samples"], sampling_params["arg_sampling"])
    print(data[0])
    save_dict[rank] = data
    return data

if __name__ == "__main__":
    test = False
    if test:
        from inspect import signature
        TInt.update_vis("__add__", CALL)
        TInt.update_vis("__sub__", CALL)
        x = TInt(942)
        y = TInt(2)
        print(getattr(TInt, "__add__"))
        print(signature(getattr(TInt, "__add__")))
        z = x.__floordiv__(y)
        print(z)
        test_ops = ["div"]#["add", "sub", "mul", "div"]
        for op in test_ops:
            res = chain_of_thought_template(op, x, y)
            resp = res["response"]
            print(res["prompt"] + resp)
            tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
            print("Pythia tok len: ", len(tok(resp).input_ids))
            tok = AutoTokenizer.from_pretrained("gpt2")
            print("gpt2 tok len: ", len(tok(resp).input_ids))
            print("\n")
        exit()

    dataset_dir = "datasets/"
    num_procs = 20

    prompt_template = chain_of_thought_template
    #prompt_template = simple_template
    # First make clean dataset
    num_train = 100000
    num_each = num_train
    train_sampling_dict = {
                            #"add": {
                            #            "num_samples": num_each,
                            #            "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                            #            "visibility": {},
                            #       },
                            #"sub": {
                            #            "num_samples": num_each,
                            #            "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                            #            "visibility": {},
                            #       },
                            #"mul": {
                            #            "num_samples": num_each,
                            #            "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
                            #            "visibility": {
                            #                            "__add__": CALL,
                            #                          },
                            #       },
                            "div": {
                                        "num_samples": num_each,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5, 1]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                        "__sub__": CALL,
                                                      },
                                   },
                          }
    test_sampling_dict = {
                            #"add": {
                            #            "num_samples": 250,
                            #            "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                            #            "visibility": {},
                            #       },
                            #"sub": {
                            #            "num_samples": 250,
                            #            "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                            #            "visibility": {},
                            #       },
                            #"mul": {
                            #            "num_samples": 250,
                            #            "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
                            #            "visibility": {
                            #                            "__add__": CALL,
                            #                          },
                            #       },
                            "div": {
                                        "num_samples": 1000,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5, 1]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                        "__sub__": CALL,
                                                      },
                                   },
                          }

    file_name = os.path.join(dataset_dir, prompt_template.__name__)
    for op_string, d in train_sampling_dict.items():
        d_train = d["arg_sampling"]
        d_test = test_sampling_dict[op_string]["arg_sampling"]
        file_name += "_{}_{}_{}_{}_{}".format(op_string, d_train[0][0], d_train[0][2], d_test[0][0], d_test[0][2])
    #file_name = dataset_dir + "{}_{}_{}_{}_{}".format(op_string, train_digit_size, test_digit_size, prompt_template.__name__, num_train_samples)
    print(f"Dumping dataset in {file_name}")

    manager = mp.Manager()
    save_dict = manager.dict()
    procs = []
    for d in train_sampling_dict.values():
        assert d["num_samples"] % num_procs == 0
        d["num_samples"] = d["num_samples"] // num_procs
    for i in range(num_procs):
        p = mp.Process(target=generate_mix_data, args=(prompt_template, train_sampling_dict, i, save_dict))
        procs.append(p)
        p.start()
        #train_clean_dataset = generate_op_data(op_string, prompt_template, num_train_samples, train_sampling)
    for p in procs:
        p.join()
    #print(train_clean_dataset)
    train_clean_dataset = [s for l in save_dict.values() for s in l]
    print("train len {}".format(len(train_clean_dataset)))
    test_clean_dataset = generate_mix_data(prompt_template, test_sampling_dict, 0, save_dict)
    print("test len {}".format(len(test_clean_dataset)))
    
    Path(file_name).mkdir(exist_ok=True, parents=True)
    dump_jsonl(os.path.join(file_name, "train.jsonl"), train_clean_dataset)
    dump_jsonl(os.path.join(file_name, "test.jsonl"), test_clean_dataset)
