import sys, trace, re, time, itertools
import torch
from util import load_jsonl, dump_jsonl, null_noise, noise_by_digit
from pathlib import Path
import os, random
from tqdm import tqdm
import numpy as np
from func import *
import operator
from transformers import AutoTokenizer
import multiprocessing as mp
from func import INVIS, VIS, CALL
import random
from functools import partial

codefile = "func.py"
f = open(codefile)
lines = f.read().splitlines()

prev_vars = {}
prev_changed_var = ''
trace = ''
exec_depth = 0
subbed_line = ""
include_code = False

num_procs = 20

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
  global codefile, prev_vars, prev_changed_var, trace, exec_depth, subbed_line, include_code
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
  #trace += "SUBBED LINE: " + subbed_line + '\n'
  return custom_trace

def no_template(op_string, noise_fn, *args):
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



def chain_of_thought_template(op_string, noise_fn, *args):
    global trace

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
    ret = ops[op_string](*args)
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
    #var_names = "xyzabcdefghijklmnopqrstuvwxyz"
    #for i in range(len(args)):
    #    trace += "{} = {}\n".format(var_names[i], args[i])
    prompt = op_to_prompt[op_string].format(*args) + "\n"
    response = trace + f"{ret}"

    prompt = insert_number_spaces(prompt)
    # Statically noise the response
    response = noise_fn(response)
    response = insert_number_spaces(response)

    # Reset global trace
    trace = ''

    reset_vars()

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

def line_noise(datapoint, linenoise):
    core = datapoint["response"].split("\n")[:-1]
    answer = datapoint["response"].split("\n")[-1]
    samples = np.random.rand(len(core))
    new_core = []
    for i, line in enumerate(core):
        if samples[i] > linenoise:
            new_core.append(line)
    datapoint["response"] = "\n".join(new_core) + "\n" + answer
    return datapoint

        
def generate_op_data(op_string, prompt_template, num_samples, arg_sampling, doc_noise, sample_noise, linenoise):
    print(f"Sampling {op_string}...")
    samples = sample_terms(arg_sampling, num_samples = num_samples)
    #print(arg_sampling, samples)
    data = []
    for sample in tqdm(samples):
        #print(sample)
        # sample whether to add noise to sample
        noise = torch.rand(1)
        if noise[0] > doc_noise:
            datapoint = prompt_template(op_string, null_noise, *sample)
        else:
            datapoint = prompt_template(op_string, sample_noise, *sample)
            datapoint = line_noise(datapoint, linenoise)
        #print(datapoint)
        data.append(datapoint)
    return data

def generate_mix_data(prompt_template, sampling_dict, rank, save_dict, doc_noise, sample_noise, linenoise=0):
    torch.random.seed()
    data = []
    for op, sampling_params in sampling_dict.items():
      for sampling_param in sampling_params:
        TInt.reset_vis()
        for f in sampling_param["visibility"]:
          TInt.update_vis(f, sampling_param["visibility"][f])
        if sampling_param.get("dynamic_noise") is not None:
          TInt.set_dynamic_noise(sampling_param["dynamic_noise"])
        if sampling_param.get("linenoise") is not None:
          linenoise = sampling_param.get("linenoise")
        data += generate_op_data(op, prompt_template, sampling_param["num_samples"], sampling_param["arg_sampling"], doc_noise, sample_noise, linenoise)
    #print(data[0])
    save_dict[rank] = data
    return data

def gen_noisy_dataset(prompt_template, doc_noise, noise_fn, char_noise_p, train_sampling_dict, test_sampling_dict):
    global num_procs
    dataset_dir = "datasets/{}/noisy/".format("median")

    #prompt_template = no_template#chain_of_thought_template
    generic_noise_function = noise_fn
    sample_noise = partial(generic_noise_function, char_noise_p) if noise_fn is not null_noise else null_noise
    #prompt_template = simple_template
    # First make clean dataset

    file_name = os.path.join(dataset_dir, prompt_template.__name__)
    for op_string, d in train_sampling_dict.items():
        d_train = d[0]["arg_sampling"]
        d_test = test_sampling_dict[op_string][0]["arg_sampling"]
        tr_string = "{}_{}_{}".format(d_train[0][0], d_train[0][1], d_train[0][2])
        te_string = "{}_{}_{}".format(d_test[0][0], d_test[0][1], d_test[0][2])
        file_name += "_{}_{}_{}_{}_{}".format(len(d_train), tr_string, te_string, d[0].get("dynamic_noise"), d[0].get('linenoise'))
        #file_name += "_{}_{}_{}_{}_{}_{}_{}_{}_{}_dynamicnoise_{}_linenoise_{}".format(op_string, include_code, len(d_train), d_train[0][0], d_train[0][1], d_train[0][2], d_test[0][0], d_test[0][1], d_test[0][2], d.get("dynamic_noise"), d.get('linenoise'))
    file_name += "_{}_{}_{}_{}".format(generic_noise_function.__name__, doc_noise, char_noise_p, include_code)
    print(f"Dumping dataset in {file_name}")

    manager = mp.Manager()
    save_dict = manager.dict()
    procs = []
    #for d in train_sampling_dict.values():
        #print(d["num_samples"], num_procs)
        #assert d["num_samples"] % num_procs == 0
    for i in range(num_procs):
        p = mp.Process(target=generate_mix_data, args=(prompt_template, train_sampling_dict, i, save_dict, doc_noise, sample_noise))
        procs.append(p)
        p.start()
        #train_clean_dataset = generate_op_data(op_string, prompt_template, num_train_samples, train_sampling)
    for p in procs:
        p.join()
    #print(train_clean_dataset)
    train_clean_dataset = [s for l in save_dict.values() for s in l]
    print("train len {}".format(len(train_clean_dataset)))
    test_clean_dataset = generate_mix_data(prompt_template, test_sampling_dict, 0, save_dict, 0.0, sample_noise)
    print("test len {}".format(len(test_clean_dataset)))
    
    Path(file_name).mkdir(exist_ok=True, parents=True)
    random.shuffle(train_clean_dataset)
    random.shuffle(test_clean_dataset)
    dump_jsonl(os.path.join(file_name, "train.jsonl"), train_clean_dataset)
    dump_jsonl(os.path.join(file_name, "test.jsonl"), test_clean_dataset)

if __name__ == "__main__":
    test_arithmetic = False
    test_sort = False
    if test_arithmetic:
        from inspect import signature
        TInt.update_vis("__add__", CALL)
        TInt.update_vis("__sub__", CALL)
        #TInt.set_dynamic_noise(0.1)
        x = torch.randint(0, 10, (6,)).tolist()
        y = torch.randint(0, 10, (3,)).tolist()
        x = TInt("".join([str(s) for s in x]))
        y = TInt("".join([str(s) for s in y]))
        #x = TInt(10719)
        #y = TInt(4623)
        #print(getattr(TInt, "__add__"))
        #print(signature(getattr(TInt, "__add__")))
        #z = x.__floordiv__(y)
        #print(z)
        TInt.update_vis("__add__", INVIS)
        TInt.update_vis("__sub__", INVIS)
        test_ops = ["gcd"]#, "sub", "mul", "div"]
        for op in test_ops:
            res = chain_of_thought_template(op,null_noise, x, y)
            resp = res["response"]
            print(res["prompt"] + resp)
            tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
            print("Pythia tok len: ", len(tok(resp).input_ids))
            tok = AutoTokenizer.from_pretrained("gpt2")
            print("gpt2 tok len: ", len(tok(resp).input_ids))
            print("\n")
        exit()
    if test_sort:
        include_code = False
        l = [199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189,188,187,186,185,184,183,182,181,180]
        tIntL = [TInt(x) for x in l]
        res = chain_of_thought_template("sort", null_noise, tIntL)
        resp = res["response"]
        print(res["prompt"] + resp)
        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        print("Pythia tok len: ", len(tok(resp).input_ids))
        tok = AutoTokenizer.from_pretrained("gpt2")
        print("gpt2 tok len: ", len(tok(resp).input_ids))
        print("\n")
        exit()


    num_train = 20000 // num_procs
    num_test = 1200

    dicts = []
    """for i in range(10):
        add_train_dict = {
            "num_samples": num_train,
            "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
            "visibility": {},
            "dynamic_noise": 0.05 * i,
        }
        add_test_dict = {
          "num_samples": num_test,
          "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
          "visibility": {},
          "dynamic_noise": 0,
        }
        dicts.append(({"add": add_train_dict}, {"add": add_test_dict}))

    mul_train_dict = {
      "num_samples": num_train,
      "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
      "visibility": {"__add__": CALL},
    }
    mul_test_dict = {
      "num_samples": num_test,
      "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
      "visibility": {"__add__": CALL},
    }"""

    gcd_train_dict = [{
      "num_samples": num_train,
      "arg_sampling": [["len", 1, 4], ["len", 1, 4]],
      "visibility": {"__add__": INVIS,
                    "__sub__": INVIS},
    }]
    gcd_test_dict = [{
      "num_samples": num_test,
      "arg_sampling": [["len", 1, 4], ["len", 1, 4]],
      "visibility": {"__add__": INVIS,
                    "__sub__": INVIS},
    }]
    
    """lens = [x+5 for x in range(25)]
    sort_train = {"sort": []}
    sort_test = {"sort": []}
    for x in lens:
      sort_train_dict = {
       "num_samples": num_train // len(lens),
        "arg_sampling": [["len", 2, 10] for i in range(x)],
        "visibility": {},
      }
      sort_test_dict = {
        "num_samples": num_test // len(lens),
        "arg_sampling": [["len", 2, 10] for i in range(x)],
        "visibility": {},
      }
      sort_train["sort"].append( sort_train_dict)
      sort_test["sort"].append( sort_test_dict)

    dicts.append((sort_train, sort_test))"""
    #dicts.append(({"add": add_train_dict}, {"add": add_test_dict}))
    #dicts.append(({"mul": mul_train_dict}, {"mul": mul_test_dict}))
    dicts.append(({"gcd": gcd_train_dict}, {"gcd": gcd_test_dict}))

    """mix_train_dict = {
                            "add": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "sub": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "mul": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                      },
                                   },
                            "div": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5, 1]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                        "__sub__": CALL,
                                                      },
                                   },
                          }
    mix_test_dict = {
                            "add": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "sub": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "mul": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                      },
                                   },
                            "div": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5, 1]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                        "__sub__": CALL,
                                                      },
                                   },
                          }"""
    #dicts.append((mix_train_dict, mix_test_dict))

    """dicts = []
    for i in range(3):
        add_train_dict = {
        "num_samples": num_train,
        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
        "linenoise": 0.05 * i + .9,
        "visibility": {},
        }
        add_test_dict = {
        "num_samples": num_test,
        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
        "visibility": {},
        }
        dicts.append(({"add": add_train_dict}, {"add": add_test_dict}))"""

    dicts = []

    lens = [x+4 for x in range(5)]
    sort_train = {"median": []}
    sort_test = {"median": []}
    for x in lens:
      sort_train_dict = {
       "num_samples": num_train // len(lens),
        "arg_sampling": [["len", 2, 8] for i in range(x)],
        "visibility": {"__floordiv__": CALL,
                       "__add__": CALL,
                       "__sub__": CALL,
                       "__mod__": CALL},
      }
      sort_test_dict = {
        "num_samples": num_test // len(lens),
        "arg_sampling": [["len", 2, 8] for i in range(x)],
        "visibility": {"__floordiv__": CALL,
                       "__add__": CALL,
                       "__sub__": CALL,
                       "__mod__": CALL},
      }
      sort_train["median"].append(sort_train_dict)
      sort_test["median"].append(sort_test_dict)

    dicts.append((sort_train, sort_test))

    prompt_templates = [chain_of_thought_template, no_template]

    d_noises = [1.0]
    char_noises = [0.25, 0.5, 0.75, 1.0]

    for prompt_template in prompt_templates:
        for doc_noise, char_noise in itertools.product(d_noises, char_noises):
            for train_samp, test_samp in dicts:
                # For now just noising add
                #if len(train_samp) == 1 and "add" in train_samp:
                gen_noisy_dataset(prompt_template, doc_noise, null_noise, char_noise, train_samp, test_samp)

