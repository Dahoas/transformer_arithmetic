import torch
from util import load_jsonl, dump_jsonl
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

def add_element_noise(ele, enl):
    e = torch.rand(1)[0]
    # Corruption randomly replaces token
    return str(torch.randint(0, 10, (1,)).item()) if e < enl else ele


def add_word_noise(number, snl):
    s = torch.rand(1)[0]
    return "".join([add_element_noise(e, enl) for e in number]) if s < snl else number


def simple_template(num1, num2, dnl=0, snl=0, enl=0):
    summed = num1 + num2
    return {"prompt": f"{num1}+{num2}=", "response": "ANSWER: " + str(summed)}


def chain_of_thought_template(num1, num2, dnl=0, snl=0, enl=0):
    # TODO add dynamic noiselevel
    str1 = str(num1)
    str2 = str(num2)
    
    # pad with 0s
    max_len = max(len(str1), len(str2))
    iter1 = str1[::-1]
    iter2 = str2[::-1]
    iter1 = iter1.ljust(max_len, "0")
    iter2 = iter2.ljust(max_len, "0")
    
    # iterate
    response = ""
    answer = ""
    carry = 0
    for i in range(max_len):
        dig1 = int(iter1[i])
        dig2 = int(iter2[i])
        sum_i = dig1 + dig2 + carry
        next_dig = sum_i % 10
        carry = sum_i // 10
        answer = str(next_dig) + answer
        response += f"Let's add the {i} digits. "
        response += f"{i} digit of {str1} = {dig1}. {i} digit of {str2} = {dig2}. Then, summed with carry, we have {sum_i}. "
        response += f"{i} digit is {next_dig}. The carry is now {carry}. "
        response += f"The result so far is {answer}. "
    if carry > 0:
        answer = str(carry) + answer
        response += f"Finally, we have leftover carry of {carry}. "

    response += f"ANSWER: {answer}"

    return {"prompt": f"{num1}+{num2}=", "response": response}


def corrupt_dataset(dataset, dnl, snl, enl):
    """
    dataset: dataset to noise
    dnl: dataset noise level. Percentage of samples in dataset with noise
    snl: sample noise level. Percentage of numbers in sample with noise
    enl: element noise level. Percentage of tokens in number with noise
    """
    for i, sample in tqdm(enumerate(dataset)):
        d = torch.rand(1)[0]
        dataset[i]["sample"] = " ".join([add_word_noise(word, snl) if word.isdigit() and d < dnl else word for word in sample["response"].split(" ")])

    return dataset


# Generate dataset summing up to ten digit numbers
def gen_noisy_dataset(prompt_template, noise_mode, dnl, snl, enl, num_samples=100000, max_num=int(1e5)):
    # TODO: Implement different noise modes (static vs. dynamic)
    # Create dataset
    terms = torch.randint(0, max_num, (num_samples, 2)).tolist()
    dataset = [prompt_template(term[0], term[1]) for term in terms]
    # Add static noise
    dataset = corrupt_dataset(dataset, dnl, snl, enl)
    return dataset


if __name__ == "__main__":
    dnls = [0.0]#np.linspace(0.1, 0.8, num=5)
    snls = [0.0]#dnls
    enls = [0.0]#dnls
    dataset_dir = "datasets/"

    prompt_template = chain_of_thought_template
    #prompt_template = simple_template
    # First make clean dataset
    dnl, snl, enl = 0.0, 0.0, 0.0
    num_samples = 100000
    print("Creating dataset with dnl: {}, snl: {}, enl: {}...".format(dnl, snl, enl))
    file_name = dataset_dir + "additions_{}_{}_{}_{}_{}".format(prompt_template.__name__, num_samples, dnl, snl, enl)
    test_dataset = gen_noisy_dataset(prompt_template=prompt_template, noise_mode="static", dnl=dnl, snl=snl, enl=enl, num_samples=1000)

    Path(file_name).mkdir(exist_ok=True, parents=True)
    dump_jsonl(os.path.join(file_name, "train.jsonl"), train_clean_dataset)
    dump_jsonl(os.path.join(file_name, "test.jsonl"), test_clean_dataset)
    # Then make dirty datasets
    for dnl in dnls:
        for snl in snls:
            for enl in enls:
                print("Creating dataset with dnl: {}, snl: {}, enl: {}...".format(dnl, snl, enl))
                file_name = dataset_dir + "additions_{}_{}_{}_{}".format(prompt_template.__name__, dnl, snl, enl)
                dataset = gen_noisy_dataset(prompt_template=prompt_template, noise_mode="static", dnl=dnl, snl=snl, enl=enl)

                Path(file_name).mkdir(exist_ok=True, parents=True)
                dump_jsonl(os.path.join(file_name, "train.jsonl"), dataset)
                dump_jsonl(os.path.join(file_name, "test.jsonl"), test_dataset)
