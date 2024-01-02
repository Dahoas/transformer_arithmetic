from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np


# Only predicts on response tokens
class MaskedSFTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.random_clause_dropout = .2
        self.agglomeration_level = 0
        self.num_omit = 8
        self.data = data
        self.tokenizer = tokenizer
        self.EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]
        self.preprocess(data, tokenizer)

    def preprocess(self, data, tokenizer):
        self.SEP = ""#" ;"
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.prompts = []
        self.responses = []
        max_length = max([len(tokenizer.encode(ele["prompt"] + self.SEP + ele["response"] + '<|endoftext|>')) for ele in tqdm(data)])
        self.max_length = max_length
        print("Max length: {}".format(max_length))

        # Data expected in prompt response pairs
        for ele in tqdm(data):
            prompt, response = ele["prompt"], ele["response"]
            prompt_encoding_len = len(tokenizer(prompt)["input_ids"])
            encodings_dict = tokenizer(prompt + self.SEP + response + '<|endoftext|>', truncation=True,
                                    max_length=max_length, padding="max_length")
            input_id = torch.tensor(encodings_dict['input_ids'])
            attn_mask = torch.tensor(encodings_dict['attention_mask'])
            label_mask = (input_id == self.EOS_ID).type(torch.int32)
            first_eos = label_mask.nonzero()
            # Skip text which has no eos token
            if len(first_eos) == 0:
                continue
            else:
                first_eos = first_eos[0, 0]
            label_mask[first_eos] = 0  # Want to predict on first eos_token
            assert(prompt_encoding_len >= 1)
            label_mask[:prompt_encoding_len] = 1  # Do not predict on prompt
            flipped_mask = 1 - label_mask
            self.input_ids.append(input_id)
            self.attn_masks.append(attn_mask)
            self.labels.append(self.input_ids[-1] * flipped_mask - 100 * label_mask)
            self.prompts.append(prompt)
            self.responses.append(response)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx], self.prompts[idx], self.responses[idx]

    def split_data(self, txt):
        return txt.splitlines()

    def sparsify(self, mode=None):
        if mode is None:
            pass
        elif mode == "random":
            sparsified_x = []
            for sample in self.data:
                prompt = sample["prompt"]
                response = sample["response"]
                sentences = self.split_data(response)
                answer = sentences[-1]
                core_response = sentences[:-1]
                rands = np.random.rand(len(core_response))
                include = (rands >= self.random_clause_dropout).nonzero()[0].tolist()
                core_response = [core_response[i] for i in include]
                core_response.append(answer)
                new_response = "\n".join(core_response)
                sparsified_x.append({"prompt": prompt, "response": new_response})
            self.data = sparsified_x
        elif mode == "sequential":
            sparsified_x = []
            #num_omit = self.num_omit
            # Take less than usual at the very first sparsification
            #self.num_omit = 14
            for sample in self.data:
                print(sample)
                prompt = sample["prompt"]
                response = sample["response"]
                sentences = self.split_data(response)
                answer = sentences[-1]
                core_response = sentences[:-1]
                '''if len(core_response) > num_omit:
                    new_response = core_response[num_omit:]
                else: 
                    new_response = []'''
                # Cut off first line and dump everything up to the next res
                new_response = "\n".join(core_response[1:])
                new_response = "res" + "res".join(new_response.split("res")[1:])
                new_response = new_response + "\n" + answer
                sparsified_x.append({"prompt": prompt, "response": new_response})
            self.data = sparsified_x
        elif mode == "agglomerative":
            if self.agglomeration_level == 0:
                for sample in self.data:
                    response = sample["response"]
                    splits = response.split(".")
                    # Want to make explicit carry over variables between computation
                    splits = [split for split in splits if "carry is now" in split or "result" in split or "ANSWER" in split]
                    response = ".".join(splits)
                    sample["response"] = response
            else:
                for sample in self.data:
                    response = sample["response"]
                    splits = response.split(".")
                    answer_split = splits[-1:]
                    splits = splits[:-1]
                    carry_splits = splits[2::4]
                    final_splits = splits[3::4]
                    splits = [sample for sublist in zip(carry_splits, final_splits) for sample in sublist] + answer_split
                    response = ".".join(splits)
                    sample["response"] = response
            self.agglomeration_level += 1
        else:
            raise ValueError("Mode {} not supported...".format(mode))

        self.preprocess(self.data, self.tokenizer)
