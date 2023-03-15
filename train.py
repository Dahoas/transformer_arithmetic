import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import argparse
from data.util import load_jsonl


def train(args):
    tok = AutoTokenizer.from_pretrained(args.tok_path)
    tok.pad_token = tok.eos_token

    data = load_jsonl(args.data_path)
    data = [sample["prompt"] for sample in data]

    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()

    train_size = int(0.98 * len(data))
    train_dataset, val_dataset = random_split(data, [train_size, len(data) - train_size])

    training_args = TrainingArguments(output_dir=args.model_path,
                                      num_train_epochs=100,
                                      logging_steps=100,
                                      save_strategy="no",
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32,
                                      warmup_steps=100,
                                      weight_decay=0.01,
                                      learning_rate=1.0e-4,
                                      save_total_limit=1,
                                      logging_dir="./logs",
                                      fp16=True,
                                      evaluation_strategy="epoch")

    def collate(samples):
        toks = tok(samples, padding="longest", truncation=True, return_tensors="pt")
        return toks

    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=collate).train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--tok_path", type=str, default="gpt2")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str)
    args = parser.parse_args()

    train(args)