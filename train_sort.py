import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import argparse
from data.util import load_jsonl
from data.data_sort import MaskedSFTDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import wandb
import yaml

model = None
EQ_TOK = None
tok = None
metric_batch_size = None
metric_max_length = None
RANK = None
sparsity_scheme = None
sparsity_modes = None

def call_model(prompts, batch_size=16, max_length=500):
    answers = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    prompt_batches = [prompts[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    for i, batch in enumerate(prompt_batches):
        batch = [torch.flip(prompt, dims=[0]) for prompt in batch]
        batch = pad_sequence(batch, batch_first=True, padding_value=tok(tok.pad_token).input_ids[0])
        batch = torch.flip(batch, dims=[1])
        batch = batch.cuda()
        output = model.generate(batch, max_length=max_length)
        output = tok.batch_decode(output)
        answer = [o.split("ANSWER: ")[-1].split("<|endoftext|>")[0] for o in output]
        answers += answer
        if i == 0 and RANK == 0:
            response_table = wandb.Table(columns=["responses", "extractions"], data=[[response, extraction] for response, extraction in zip(output, answer)])
            wandb.log({"generations": response_table})
    return answers

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
    responses = [r.split("ANSWER: ")[-1].split("<|endoftext|>")[0] for r in responses]
    outputs = call_model(prompts, batch_size=metric_batch_size, max_length=metric_max_length)
    is_correct = [outputs[i] == responses[i] for i in range(len(outputs))]
    return {"accuracy": sum(is_correct)/len(is_correct)}


class SparsityCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, train_dataloader, **kwargs):
        print("Entering callback...")
        epoch = state.epoch
        dataset = train_dataloader.dataset
        if sparsity_scheme is not None and RANK == 0 and len(sparsity_scheme) > 0 and epoch == int(sparsity_scheme[0]): 
            print("Current sparsity scheme:")
            print(sparsity_modes)
            print(sparsity_scheme)
            # Sparsification epoch reached
            mode = sparsity_modes.pop(0)
            sparsity_scheme.pop(0)
            print("Sparsifying at epoch {} with model {}...".format(epoch, mode))
            print("Before sparsification: {}".format(dataset.data[0]["response"]))
            dataset.sparsify(mode)
            print("After sparsification: {}".format(dataset.data[0]["response"]))
                



def train(args):
    global EQ_TOK, tok, model, metric_batch_size, metric_max_length, RANK, sparsity_scheme, sparsity_modes

    RANK = args.local_rank
    sparsity_modes = args.sparsity_modes
    sparsity_scheme = args.sparsity_scheme

    metric_batch_size = args.metric_batch_size
    tok = AutoTokenizer.from_pretrained(args.tok_path)
    EQ_TOK = tok("=").input_ids[0]
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()

    train_data = load_jsonl(os.path.join(args.data_path, "train.jsonl")) if args.train_data_size is None else load_jsonl(os.path.join(args.data_path, "train.jsonl"))[:args.train_data_size]
    val_data = load_jsonl(os.path.join(args.data_path, "test.jsonl"))[:args.metric_data_size]
    train_dataset = MaskedSFTDataset(train_data, tok)
    val_dataset = MaskedSFTDataset(val_data, tok)
    metric_max_length = val_dataset.max_length + 25

    batch_size = args.batch_size
    steps_per_epoch = len(train_dataset) // (batch_size * torch.cuda.device_count())
    eval_steps = steps_per_epoch

    if int(args.local_rank) == 0:
        wandb.init(project="transformer_arithmetic", config=vars(args))

    training_args = TrainingArguments(output_dir="out/",
                                      num_train_epochs=args.epochs,
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
                                      eval_steps=eval_steps,
                                      eval_accumulation_steps=2,
                                      include_inputs_for_metrics=True,
                                      fp16_full_eval=True,
                                      gradient_checkpointing=bool(args.gradient_checkpointing),
                                    )

    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]),'labels': torch.stack([f[2] for f in data])}

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, compute_metrics=compute_metrics,
            eval_dataset=val_dataset, data_collator=data_collator, callbacks=[SparsityCallback])
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train_data_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--metric_data_size", type=int, default=100)
    parser.add_argument("--metric_batch_size", type=int, default=16)
    parser.add_argument("--tok_path", type=str, default="gpt2")
    parser.add_argument("--sparsity_scheme", nargs="*", default=None)
    parser.add_argument("--sparsity_modes", nargs="*", default=None)
    parser.add_argument("--gradient_checkpointing", type=int, default=0)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str)
    args = parser.parse_args()

    assert (args.sparsity_scheme is None and args.sparsity_modes is None) or len(args.sparsity_scheme) == len(args.sparsity_modes)

    if args.config_path is not None:
        config = yaml.safe_load(open(args.config_path, "r"))
        for key, val in config.items():
            setattr(args, key, val)

    # Parsing data_path for logging purposes
    data_args = os.path.basename(args.data_path).split("_")
    task = data_args.pop(0)
    dnl, snl, enl = data_args[-3:]
    prompt_template = "_".join(data_args[:-3])
    args.task = task
    args.dnl = dnl
    args.snl = snl
    args.enl = enl
    args.prompt_template = prompt_template

    train(args)
