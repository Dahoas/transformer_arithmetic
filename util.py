import json
from tqdm import tqdm
import torch
from datasets import load_dataset


######## Data I/O Utilities ########

def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = json.loads(line)
            data.append(response)
    return data


def dump_jsonl(filename, data):
    with open(filename, "w") as f:
        for dict_t in data:
                json.dump(dict_t, f)
                f.write("\n")


def str_to_int(x):
    return int("".join(x.split(" ")))


def check_data(data_path):
    data = load_jsonl(data_path)
    clean = 0
    for sample in tqdm(data):
        x, y = sample["prompt"].split(" + ")
        x = str_to_int(x)
        y = str_to_int(y)
        r = sample["response"].split("\n")[-1]
        r = int("".join(r.split(" ")))
        clean += int(r == x + y)
    print(clean)