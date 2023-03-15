import torch
from util import load_jsonl, dump_jsonl


def simple_template(num1, num2, noise_level):
    summed = num1 + num2
    if torch.rand(1).item() < noise_level:
        summed += torch.randint(-int(summed**0.5), int(summed**0.5), (1,)).item()
    return f"{num1} + {num2} = {summed}"


# Generate dataset summing up to ten digit numbers
def gen_noisy_dataset(noise_level, file_name, prompt_template, num_samples=100000, max_num=int(1e10)):
    terms = torch.randint(0, max_num, (num_samples, 2)).tolist()
    dataset = []
    for term in terms:
        prompt = {"prompt": prompt_template(term[0], term[1], noise_level)}
        dataset.append(prompt)
    dump_jsonl(file_name, dataset)


if __name__ == "__main__":
    noise_levels = [0.0, 0.05, 0.2]
    prompt_template = simple_template
    for noise_level in noise_levels:
        file_name = "additions_{}_{}.jsonl".format(noise_level, prompt_template.__name__)
        gen_noisy_dataset(noise_level=noise_level, file_name=file_name, prompt_template=prompt_template)