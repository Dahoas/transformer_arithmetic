import torch
from util import load_jsonl, dump_jsonl
from pathlib import Path
import os
from tqdm import tqdm


def add_measurement_error(noise_level, noise_magnitude, num1, num2):
    summed = num1 + num2
    if torch.rand(1).item() < noise_magnitude:
        summed += torch.randint(-int(summed**0.5), int(summed**0.5), (1,)).item()
    return summed


def add_element_noise(ele, enl):
    e = torch.rand(1)[0]
    # Corruption randomly replaces token
    return str(torch.randint(0, 10, (1,)).item() % 10) if e < enl else ele


def add_word_noise(number, snl):
    s = torch.rand(1)[0]
    return "".join([add_element_noise(e, enl) for e in number]) if s < snl else number


def simple_template(num1, num2, noise_level):
    summed = num1 + num2
    if torch.rand(1).item() < noise_level:
        summed += torch.randint(-int(summed**0.5), int(summed**0.5), (1,)).item()
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
        response += f"Finally, we have leftover carry of {carry}. ANSWER: {answer}"

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
    dataste = corrupt_dataset(dataset, dnl, snl, enl)
    return dataset


if __name__ == "__main__":
    #noise_levels = [0.0]#[0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    #dnls = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    #snls = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    #enls = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    dnls = [0.001, 0.01, 0.1]
    snls = [0.001, 0.01, 0.1]
    enls = [0.001, 0.01, 0.1]
    dataset_dir = "datasets/"

    prompt_template = chain_of_thought_template
    # First make clean dataset
    dnl, snl, enl = 0.0, 0.0, 0.0
    print("Creating dataset with dnl: {}, snl: {}, enl: {}...".format(dnl, snl, enl))
    file_name = dataset_dir + "additions_{}_{}_{}_{}".format(prompt_template.__name__, dnl, snl, enl)
    train_clean_dataset = gen_noisy_dataset(prompt_template=prompt_template, noise_mode="static", dnl=dnl, snl=snl, enl=enl)
    test_clean_dataset = gen_noisy_dataset(prompt_template=prompt_template, noise_mode="static", dnl=dnl, snl=snl, enl=enl, num_samples=1000)

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
                dump_jsonl(os.path.join(file_name, "test.jsonl"), test_clean_dataset)
