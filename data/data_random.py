from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

random_clause_dropout = .2

def sparsify(x, mode):
    sparsified_x = []
    for sample in x:
        prompt = sample["prompt"]
        response = sample["response"]
        if mode == "random":
            # sentences
            sentences = response.split(". ")
            suffix = sentences[-1]
            core_response = sentences[:-1]
            rands = np.random.rand(len(core_response))
            include = (rands >= random_clause_dropout).nonzero()[0].tolist()
            core_response = [core_response[i] for i in include]
            core_response.append(suffix)
            new_response = ". ".join(core_response)
        elif mode == "sequential":
            split_string = "The carry is now"
            index = response[1:].find(split_string) + 1
            
            new_response = response
            if index > 0:
                new_response = response[index:]
            else: new_response = response[response.find("ANSWER: "):]
        sparsified_x.append({"prompt": prompt, "response": new_response})
    return (sparsified_x)


test_data = [{"prompt": "52+389=", "response": "Let's add the 0 digits. 0 digit of 52 = 2. 0 digit of 389 = 9. Then, summed with carry, we have 11. 0 digit is 1. The carry is now 1. The result so far is 1. Let's add the 1 digits. 1 digit of 52 = 5. 1 digit of 389 = 8. Then, summed with carry, we have 14. 1 digit is 4. The carry is now 1. The result so far is 41. Let's add the 2 digits. 2 digit of 52 = 0. 2 digit of 389 = 3. Then, summed with carry, we have 4. 2 digit is 4. The carry is now 0. The result so far is 441. ANSWER: 441"}, {"prompt": "84+24=", "response": "Let's add the 0 digits. 0 digit of 84 = 4. 0 digit of 24 = 4. Then, summed with carry, we have 8. 0 digit is 8. The carry is now 0. The result so far is 8. Let's add the 1 digits. 1 digit of 84 = 8. 1 digit of 24 = 2. Then, summed with carry, we have 10. 1 digit is 0. The carry is now 1. The result so far is 08. Finally, we have leftover carry of 1. ANSWER: 108"}]


for i in range(100):
    print(test_data)
    test_data = sparsify(test_data, "random")

print(test_data)
