import json
from tqdm import tqdm
import torch
from datasets import load_dataset


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



##############Noise Patterns############

def null_noise(num, *args, **kwargs):
    return num

def noise_by_digit(c_noise_p, num):
    # choose whether to noise number
    s = str(num)
    s = s.split("\n")
    t = []
    for line in s:
        l = []
        for word in line.split(" "):
            w_res = ''
            if word.isnumeric():
                for c in word:
                    p_corrupt_char = torch.rand(1)
                    if p_corrupt_char < c_noise_p:
                        w_res += str(torch.randint(0, 10, (1,)).item())
                    else:
                        w_res += c
            else:
                w_res = word
            l.append(w_res)
        t.append(" ".join(l))
    t = "\n".join(t)
    return t

###########Dump Dataset###########
def anthropic_hh_dump():
    data = load_dataset("Dahoas/full-hh-rlhf")["train"]
    print(len(data))
    print(data[0])
    dump_jsonl("train.jsonl", data)
    


if __name__ == "__main__":
    data_path = "datasets/noisy_datasets/no_template_add_len_1_10_len_1_10_noise_by_digit_0.8_0.5/train.jsonl"
    #check_data(data_path)
    anthropic_hh_dump()

