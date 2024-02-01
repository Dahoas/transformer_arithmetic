import argparse
import numpy as np
import json
import os
import pathlib
from tqdm import tqdm

from util import load_jsonl


def evaluate(dataset_path):
    results = {}
    dataset = load_jsonl(dataset_path)

    def score_sample(sample):
        gt_answer = int(sample["response"].split("\n")[-1].strip().replace(" ", ""))
        try:
            model_answer = int(sample["model_response"].split("\n")[-1].split("=")[-1].strip().replace(" ", ""))
            parse_error = 0
        except Exception:
            model_answer = -1
            parse_error = 1
        return {"score": int(model_answer == gt_answer), "parse_error": parse_error}

    scores = [score_sample(sample) for sample in dataset]
    results["accuracy"] = np.mean([res["score"] for res in scores])
    results["parse_error"] = np.mean([res["parse_error"] for res in scores])

    print(f"{dataset_path} results...")
    print(json.dumps(results, indent=2))
    op = os.path.basename(os.path.dirname(dataset_path))

    pathlib.Path("results").mkdir(exist_ok=True, parents=True)
    with open(f"results/{op}_{os.path.basename(dataset_path)}", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_folder", type=str, default=None, help="Path to folder containing datasets to evaluate")
    args = parser.parse_args()

    args_dict = vars(args)

    dataset_folder = args_dict.pop("dataset_folder")
    if dataset_folder is not None:
        dataset_paths = pathlib.Path(dataset_folder).glob("*")
    else:
        dataset_paths = [args.dataset_path]
    
    for dataset_path in tqdm(dataset_paths):
        evaluate(dataset_path=dataset_path)