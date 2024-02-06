import argparse
from typing import List
import os
import random

from gptquery.gpt import GPT
from gptquery.logger import Logger

from util import load_jsonl


"""
Experiment design:
- use gpt-4 (gpt-3.5 can't do it)
to minimize cost use num_test = 100

Do noise experiments on
- add 1-10
- med 1-8
- mult 1-8
with three levels of dataset noise
- 0.0, 0.2, 0.5, 0.8
    - Use 0.3, 0.6, 0.9 char, line
    - 0.2, 0.4 dynamic

3 + 3 * 3 * 3 + 3 * 3 * 2 = 58
- 5800 total queries -> 5800 minutes

Run 0.5 noise first to see how that affects the model

Also test on sorting and gcd tasks with clean data
"""


def make_few_shot_prompt(exemplars, 
                         num_few_shot,):
    random.shuffle(exemplars)
    exemplars = [f"{sample.get('prompt')}{sample.get('response')}" for sample in exemplars][:num_few_shot]
    return "\n\n".join([f"Q{i}: {exemplar}" for i, exemplar in enumerate(exemplars)]) + f"Q{len(exemplars)}: " + "{prompt}"


def solve(exemplar_path, 
          num_few_shot,
          dataset_path, 
          save_folder,
          model_name,
          oai_key,):
    system_prompt_text = "Below are several questions and solutions solved using an algorithm. Use this algorithm to solve the last question. \
Do NOT use code. Make sure to give your final answer on the last newline without any other text."
    exemplars = load_jsonl(exemplar_path)
    task_prompt_text = make_few_shot_prompt(exemplars, num_few_shot)

    gpt = GPT(model_name=model_name,
              system_prompt_text=system_prompt_text,
              task_prompt_text=task_prompt_text,
              oai_key=oai_key,
              mb_size=1,
              temperature=0.0,
              verbose=True,)
    model_name = model_name.replace("-", "_")

    dataset = load_jsonl(dataset_path)
    # Extract op from dataset_path
    op = os.path.basename(os.path.dirname(os.path.dirname(dataset_path)))
    params = os.path.basename(os.path.dirname(dataset_path))
    Logger.init(os.path.join(save_folder, op, f"{params}_{model_name}_num_few_shot_{num_few_shot}.jsonl"))
    output = gpt(dataset, output_key="model_response")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exemplar_path", type=str)
    parser.add_argument("--num_few_shot", type=int, default=6)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_folder", default="datasets/rollouts/")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-1106")
    parser.add_argument("--oai_key", type=str)
    args = parser.parse_args()

    solve(**vars(args))