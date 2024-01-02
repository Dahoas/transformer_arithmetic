import argparse
from typing import List

from gptquery.gpt import GPT
from gptquery.logger import Logger

from util import load_jsonl


def make_few_shot_prompt(exemplars):
    return "\n\n".join([f"Q{i}: {exemplar}" for i, exemplar in enumerate(exemplars)]) + f"Q{len(exemplars)}: " + "{question}"


def solve(exemplar_path, dataset_path):
    system_prompt_text = "Solve the following questions using the provided algorithm."
    exemplars = load_jsonl(exemplar_path)
    task_prompt_text = make_few_shot_prompt(exemplars)

    gpt = GPT(model_name="gpt-3.5-1106-preview",
              system_prompt_text=system_prompt_text,
              task_prompt_text=task_prompt_text,
              oai_key=oai_key,
              mbs=1,
              temperature=0.0,)

    dataset = load_jsonl(dataset_path)
    Logger.init("temp.jsonl")
    output = gpt(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exemplar_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--oai_key", type=str)
    args = parser.parse_args()

    oai_key = args.oai_key

    solve(**vars(args))