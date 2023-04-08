import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import argparse
from data.util import load_jsonl
from data.data import MaskedSFTDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os

model = None
EQ_TOK = None
tok = None

def call_model(prompts, batch_size=4, max_length=500):
    outputs = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    prompt_batches = [prompts[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    for batch in prompt_batches:
        batch = [torch.flip(prompt, dims=[0]) for prompt in batch]
        batch = pad_sequence(batch, batch_first=True)
        batch = torch.flip(batch, dims=[1])
        batch = batch.cuda()
        output = model.generate(batch, max_length=max_length)
        output = tok.batch_decode(output)
        output = [o.split("=")[1].split("ANSWER: ")[-1].replace("<|endoftext|>","") for o in output]
        outputs += output
    return outputs

# TODO: retrieve prompt, response AND FINAL ANSWER
# TODO: maybe need to have a standard way to end the prompt

def compute_metrics(eval_preds):
    preds = eval_preds.predictions
    inputs = torch.tensor(eval_preds.inputs)
    split_inds = torch.argmax((inputs == EQ_TOK).type(torch.float32), dim=1).flatten()
    prompts = []
    responses = []
    for split_ind, inp in zip(split_inds, inputs):
        #  1. Retrieve prompt and response
        #  2. Call model on prompt
        #  3. Compare model ouptut to response
        prompt = inp[:split_ind+1]
        response = inp[split_ind+1:]
        prompts.append(prompt)
        responses.append(response)
    responses = tok.batch_decode(responses)
    responses = [r.split("ANSWER: ")[-1].replace("<|endoftext|>", "") for r in responses]
    outputs = call_model(prompts)
    is_correct = [outputs[i] == responses[i] for i in range(len(outputs))]
    return {"accuracy": sum(is_correct)/len(is_correct)}

def train(args):
    global EQ_TOK, tok, model

    tok = AutoTokenizer.from_pretrained(args.tok_path)
    EQ_TOK = tok("=").input_ids[0]
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()


    train_data = load_jsonl(os.path.join(args.data_path, "train.jsonl"))
    val_data = load_jsonl(os.path.join(args.data_path, "test.jsonl"))
    train_dataset = MaskedSFTDataset(train_data, tok)
    val_dataset = MaskedSFTDataset(val_data, tok)

    batch_size = 20
    steps_per_epoch = len(train_dataset) // (batch_size * torch.cuda.device_count())

    training_args = TrainingArguments(output_dir="out/",
                                      num_train_epochs=200,
                                      logging_steps=100,
                                      save_strategy="no",
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      warmup_steps=100,
                                      weight_decay=0.01,
                                      learning_rate=1.0e-4,
                                      save_total_limit=1,
                                      logging_dir="./logs",
                                      fp16=True,
                                      evaluation_strategy="steps",
                                      eval_steps=4*steps_per_epoch,
                                      include_inputs_for_metrics=True)

    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]),'labels': torch.stack([f[2] for f in data])}

    Trainer(model=model, args=training_args, train_dataset=train_dataset, compute_metrics=compute_metrics,
            eval_dataset=val_dataset, data_collator=data_collator).train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="gpt2-large")
    parser.add_argument("--tok_path", type=str, default="gpt2")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str)
    args = parser.parse_args()

    train(args)
