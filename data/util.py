import json


def load_jsonl(filename):
	data = []
	with open(filename, "r") as f:
		lines = f.readlines()
		for line in lines:
			response = json.loads(line)
			data.append(response)
	return data

def dump_jsonl(filename, data):
	with open(filename, "w") as f:
		for dict_t in data:
				json.dump(dict_t, f)
				f.write("\n")


def check_data(data_path):
	return None


if __name__ == "__main__":
	data_path = "datasets/additions_chain_of_thought_template_0.0_0.0_0.0/train.jsonl"
	check_data(data_path)