import json
import random


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(result_file_name):
    results = []
    with open(result_file_name, "r") as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)
    return results


def append_to_jsonl(file_path, data):
    with open(file_path, "a") as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + "\n")


def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()
    return content.split("\n\n")


def generate_inputs(data, instructions):
    inputs = []
    for item in data:
        instruction = random.choice(instructions)
        inputs.append(
            {
                "input": f"{instruction}\n\n{item['input']}".strip("\n"),
                "task": item["task"],
                "id": item["id"],
            }
        )
    return inputs


def get_model_name(model_path):
    model_name = model_path.split("/")[-1]
    if model_name.startswith("checkpoint-"):
        model_name = model_path.split("/")[-2] + "_" + model_name
    return model_name
