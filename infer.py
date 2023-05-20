import torch
from data.util import load_jsonl
from data.data import MaskedSFTDataset
from accelerate import Accelerator
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re


def model_with_call(model, tok, tok_prompts, max_length):
    while True:
        output = model.generate(tok_prompts, max_length=max_length)
        text_output = tok.batch_decode(output)[0]
        if len(output) == max_length:
            break
        elif "call(" in text_output:
            res = re.findall(r"call\( (.*) \)", text_output)[0]+"\n"
            tok_res = tok(res, return_tensors="pt").input_ids.to(tok_prompts.device)
            sub_out = model_with_call(model, tok, tok_res, max_length)
            prev_call = text_output.split(f"call( {res} )")[0]
            next_prompt = prev_call_line + res + prev_call.split("\n")[-1].split("= ")[0] + "= " + sub_out.split("<|endoftext|>")[0].split("\n")[-1]
            tok_prompts = tok(next_prompt, return_tensors="pt")[0]
        elif "<|endoftext|>" in text_output:
            break
    return text_output


def infer(model, dataloader, tokenizer, max_length, temp):
    """Function to infer causal model in parallel on dataloader 
    with at most max_length tokens at temperature temp.
    """
    scores = []
    for inputs in tqdm(dataloader):
        prompts, responses = inputs["prompts"], inputs["responses"]
        tok_prompts = tokenizer(prompts, return_tensors="pt").input_ids.to(accelerator.device)
        outputs = model.generate(tok_prompts, max_length=max_length, do_sample=temp>0, temperature=temp)
        text_outputs = tokenizer.batch_decode(outputs)
        model_answers = [s.split("<|endoftext|>")[0].split("\n")[-1] for s in text_outputs]
        gt_answers = [response.split("\n")[-1] for response in responses]
        scores += [int(model_answer == gt_answer) for model_answer, gt_answer in zip(model_answers, gt_answers)]
    scores = torch.tensor(scores, device=tok_prompts.device)
    torch.distributed.all_reduce(scores)
    num_correct = scores.sum().item() / (torch.cuda.device_count() * len(scores))
    if torch.distributed.get_rank() == 0:
        print(f"Score: {num_correct}")


if __name__ == "__main__":
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
    dataset = load_jsonl(args.prompt_dataset)[:args.test_size]
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