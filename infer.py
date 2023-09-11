import torch
from data.util import load_jsonl
from accelerate import Accelerator
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import re
import random

# custom stopping criteria STOPS when last line has a call( * )
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, tok = None):
        self.tok = tok
        return
    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        toks = input_ids[0]
        text_output = self.tok.decode(toks)
        lines = text_output.splitlines()
        last_line_res = re.findall(r"call\( (.*) \)", lines[-1])
        return len(last_line_res) > 0 or "<|endoftext|>" in text_output

stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria()])

#TODO: 1. Stop generation when </call> is made to speed up eval.
###### 2. Sync subcalled generation across all gpus with an allreduce. Necessary for parallel eval
def subcall_model(model, tok, tok_prompts, max_length):
    global stopping_criteria
    assert len(tok_prompts) == 1
    while True:
        #if torch.distributed.get_rank() == 0:
        #print("Calling model with <<", tok.batch_decode(tok_prompts)[0], ">> on rank ", torch.distributed.get_rank())
        output = model.generate(tok_prompts, max_length=max_length, stopping_criteria=stopping_criteria)
        text_output = tok.batch_decode(output)
        #print("Model ", torch.distributed.get_rank(), " response: ", text_output[0])
        if len(output[0]) == max_length:
            break
        elif "<|endoftext|>" in text_output[0]:
            break
        elif "call(" in text_output[0]:
            text_output = text_output[0]
            res = re.findall(r"call\( (.*) \)", text_output)[-1]+"\n"
            tok_res = tok(res, return_tensors="pt").input_ids.to(tok_prompts.device)
            sub_out = subcall_model(model, tok, tok_res, max_length)[0]
            next_prompt = text_output + "\n" + text_output.split("\n")[-1].split("= ")[0] + "= " + sub_out.split("<|endoftext|>")[0].split("\n")[-1] + "\n"
            tok_prompts = tok(next_prompt, return_tensors="pt").input_ids.to(tok_prompts.device)
    return text_output


def infer(model, dataloader, tokenizer, max_length, temp):
    """Function to infer causal model in parallel on dataloader 
    with at most max_length tokens at temperature temp.
    """
    scores = []
    for inputs in tqdm(dataloader):
        prompts, responses = inputs["prompts"], inputs["responses"]
        tok_prompts = tokenizer(prompts, return_tensors="pt").input_ids.to(accelerator.device)
        #text_outputs = tokenizer.batch_decode(outputs)
        text_outputs = subcall_model(model, tokenizer, tok_prompts, max_length)
        model_answers = [s.split("<|endoftext|>")[0].split("\n")[-1] for s in text_outputs]
        gt_answers = [response.split("\n")[-1] for response in responses]
        scores += [int(model_answer == gt_answer) for model_answer, gt_answer in zip(model_answers, gt_answers)]
    scores = torch.tensor(scores, device=tok_prompts.device)
    torch.distributed.all_reduce(scores)
    num_correct = scores.sum().item() / (torch.cuda.device_count() * len(scores))
    if torch.distributed.get_rank() == 0:
        print(f"Score: {num_correct}")


if __name__ == "__main__":
    # Command to run: accelerate launch --num_procs 1 infer.py 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset", type=str)
    parser.add_argument("--model_path", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--tok_path", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--temp", default=0, type=float)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--test_size", default=104, type=int)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tok_path)
    tok.pad_token = tok.eos_token
    stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(tok)])

    dataset = load_jsonl(args.prompt_dataset)[:args.test_size]
    random.shuffle(dataset)
    assert len(dataset) % torch.cuda.device_count() == 0
    dataset = MaskedSFTDataset(dataset, tok)
    args.max_length = dataset.max_length + 25

    prompt_dataset = dataset
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()
    model.half()

    data_collator = lambda data: {
                                    'prompts': [f[3] for f in data], 
                                    'responses': [f[4] for f in data]
                                 }
    dataloader = torch.utils.data.DataLoader(prompt_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    model = accelerator.unwrap_model(model)
    model = model.to(accelerator.device)

    infer(model, dataloader, tok, args.max_length, args.temp)
