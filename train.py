import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import argparse
from data.util import load_jsonl
from data.data import MaskedSFTDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import wandb
import yaml, random


model = None
tok = None
metric_batch_size = None
metric_max_length = None
RANK = None
sparsity_scheme = None
sparsity_modes = None
val_dataset = None

def call_model(prompts, gts, val_answers, batch_size=16, max_length=500):
    answers = []
    inputs = []
    responses = []
    # Batch prompts over gpus
    prompts_per_gpu = (len(prompts) + torch.cuda.device_count() - 1) // torch.cuda.device_count()
    prompts = prompts[RANK * prompts_per_gpu : (RANK + 1) * prompts_per_gpu]
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    prompt_batches = [prompts[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    for i, batch in enumerate(prompt_batches):
        batch = [torch.flip(prompt, dims=[0]) for prompt in batch]
        batch = pad_sequence(batch, batch_first=True, padding_value=tok(tok.pad_token).input_ids[0])
        batch = torch.flip(batch, dims=[1])
        #print(RANK, tok.decode(batch[0]))
        #exit()
        batch = batch.to(f"cuda:{RANK}")
        output = model.generate(batch, max_length=max_length)
        output = tok.batch_decode(output)
        #print(RANK, output)
        answer = ["".join((o.split("<|endoftext|>")[0].split("\n")[-1]).split()) for o in output]
        answers += answer
        inputs += tok.batch_decode(batch)
        responses += output
    if RANK == 0:
        response_table = wandb.Table(columns=["inputs", "responses", "extractions", "gts", "val_answers"], data=[[inp, response, extraction, gt, val_answer] for inp, response, extraction, gt, val_answer in zip(inputs, responses, answers, gts, val_answers)])
        wandb.log({"generations": response_table})
    return answers

def compute_metrics(eval_preds):
    #inputs = torch.tensor(eval_preds.inputs)
    #TODO(dahoas): Generalize from equality token to split prompt and response
    prompts = val_dataset.prompts
    prompts = [tok(prompt, return_tensors="pt").input_ids[0] for prompt in prompts]
    prompts_per_gpu = (len(prompts) + torch.cuda.device_count() - 1) // torch.cuda.device_count()
    responses = val_dataset.responses[RANK * prompts_per_gpu : (RANK + 1) * prompts_per_gpu]
    answers = ["".join((r.split("\n")[-1]).split()) for r in responses]
    outputs = call_model(prompts, responses, answers, batch_size=metric_batch_size, max_length=metric_max_length)
    is_correct = torch.tensor([int(outputs[i] == answers[i]) for i in range(len(outputs))], device=f"cuda:{RANK}")
    #print("\n\n\nRANK=0!", is_correct)
    assert len(prompts) % torch.cuda.device_count() == 0
    torch.distributed.all_reduce(is_correct)
    #if RANK == 0:
    #    print("\n\n\nTOTAL", is_correct)
    num_correct = is_correct.sum().item() / len(prompts)
    return {"accuracy": num_correct}


class SparsityCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, train_dataloader, **kwargs):
        print("Entering callback...")
        epoch = state.epoch
        dataset = train_dataloader.dataset
        # Should this actually be done on all ranks? After all doesn't each rank have its own copy of the data
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
    global tok, model, metric_batch_size, metric_max_length, RANK, sparsity_scheme, sparsity_modes, val_dataset

    RANK = args.local_rank
    sparsity_modes = args.sparsity_modes
    sparsity_scheme = args.sparsity_scheme

    metric_batch_size = args.metric_batch_size
    tok = AutoTokenizer.from_pretrained(args.tok_path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()

    train_data = load_jsonl(os.path.join(args.data_path, "train.jsonl")) if args.train_data_size is None else load_jsonl(os.path.join(args.data_path, "train.jsonl"))
    train_data = train_data[:args.train_data_size]
    val_data = load_jsonl(os.path.join(args.data_path, "test.jsonl"))
    val_data = val_data[:args.metric_data_size]
    #val_data = load_jsonl(os.path.join(args.data_path, "test.jsonl"))[:args.metric_data_size]
    train_dataset = MaskedSFTDataset(train_data, tok)
    val_dataset = MaskedSFTDataset(val_data, tok)
    # TODO(dahoas): Adjsut metric_max_length to generalize outside of addition
    metric_max_length = val_dataset.max_length + 25

    batch_size = args.batch_size
    steps_per_epoch = len(train_dataset) // (batch_size * torch.cuda.device_count())
    eval_steps = steps_per_epoch

    if int(args.local_rank) == 0:
        wandb.init(project="transformer_arithmetic", config=vars(args))

    training_args = TrainingArguments(output_dir="out1/",
                                      num_train_epochs=args.epochs,
                                      logging_steps=20,
                                      save_strategy="no",
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=metric_batch_size,
                                      warmup_steps=100,
                                      weight_decay=0.01,
                                      learning_rate=1.0e-4,
                                      # save_total_limit=2,
                                      logging_dir="./logs",
                                      fp16=True,
                                      evaluation_strategy="steps",
                                      eval_steps=eval_steps,
                                      save_steps=eval_steps, #test
                                      eval_accumulation_steps=2,
                                      include_inputs_for_metrics=True,
                                      fp16_full_eval=True,
                                      gradient_checkpointing=bool(args.gradient_checkpointing),
                                    )

    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]),'labels': torch.stack([f[2] for f in data])}

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, compute_metrics=compute_metrics,
            eval_dataset=val_dataset, data_collator=data_collator, callbacks=[SparsityCallback])
    trainer.train()
    #trainer.save_model('{}_{}_{}'.format(args.data_path, args.train_data_size, args.epochs))
    model.save_pretrained('ckpts/{}_{}_{}'.format(args.model_path, args.data_path, args.epochs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--tok_path", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train_data_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--metric_data_size", type=int, default=104)
    parser.add_argument("--metric_batch_size", type=int, default=4)
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
    dnl, snl, enl = 0, 0, 0#data_args[-3:]
    prompt_template = "_".join(data_args[:-3])
    args.task = task
    args.dnl = dnl
    args.snl = snl
    args.enl = enl
    args.prompt_template = prompt_template
    #args.sparsity_modes = [(i, "random") for i in range(5)]
    #args.sparsity_scheme = [(0, 2), (1, 10), (2, 25), (3, 45), (4, 70)]
    train(args)
