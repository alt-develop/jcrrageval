import dotenv

dotenv.load_dotenv()

import json
import argparse
from openai import OpenAI
import os
from collections import defaultdict
from rag_eval import llm_judge as llm_judge_rag, auto_metrics
from hallucination_eval import llm_judge as llm_judge_hallucination
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path



def get_model_name(model_path):
    model_name = model_path.split("/")[-1]
    if model_name.startswith("checkpoint-"):
        model_name = model_path.split("/")[-2] + "_" + model_name
    return model_name


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def append_to_jsonl(file_path, data):
    with open(file_path, "a") as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            
def auto_evaluate(model, benchmark_name):
    data = read_json(f"../data/{benchmark_name}.json")
    task_index_map = defaultdict(list)
    for i, item in enumerate(data):
        task_index_map[item["task"]].append(i)

    model_name = get_model_name(model)
    model_answer_file = (
        f"../gen_answer/model_answer/{benchmark_name}/{model_name}.json"
    )
    if not os.path.exists(model_answer_file):
        print(f"Model answer file not found for {model_name}. Skipping.")
        return None
    response = read_json(model_answer_file)

    os.makedirs(f"model_judgement/{benchmark_name}", exist_ok=True)
    os.makedirs(f"auto_metrics/{benchmark_name}", exist_ok=True)

    if benchmark_name != "wikirag":
        rag_data = [data[i] for i in task_index_map["rag"]]
        hallucination_data = [data[i] for i in task_index_map["hallucination"]]

    else:
        # rag: 1, 2
        # hallucination: 3, 4
        rag_data = [data[i] for i in task_index_map[1] + task_index_map[2]]
        hallucination_data = [
            data[i] for i in task_index_map[3] + task_index_map[4]
        ]
    
    print("Benchmark name: ", benchmark_name)
    print("RAG data: ", len(rag_data))
    print("Hallucination data: ", len(hallucination_data))


    if rag_data:
        auto_metrics_result = auto_metrics(data, response)
        auto_metrics_result["model"] = model_name
        auto_metrics_result["task"] = "rag"
        auto_metrics_result.to_json(
            f"auto_metrics/{benchmark_name}/{model_name}.jsonl",
            orient="records", lines=True, force_ascii=False
        )
        return auto_metrics_result
        
    return None
    

def evaluate_models(judger, model, benchmark_name, rag_scale=5):
    data = read_json(f"../data/{benchmark_name}.json")
    data_by_id = {str(item['id']): item for item in data}
    
    task_index_map = defaultdict(list)
    for id, sample in data_by_id.items():
        task_index_map[sample["task"]].append(str(id))

    model_name = get_model_name(model)
    model_answer_file = (
        f"../gen_answer/model_answer/{benchmark_name}/{model_name}.json"
    )
    if not os.path.exists(model_answer_file):
        print(f"Model answer file not found for {model_name}. Skipping.")
        return None
    response = read_json(model_answer_file)
    response_by_id = {str(item['id']): item for item in response}

    os.makedirs(f"model_judgement/{benchmark_name}", exist_ok=True)
    os.makedirs(f"auto_metrics/{benchmark_name}", exist_ok=True)
    
    assert len(data_by_id) == len(response_by_id), f"Data and response length mismatch: {len(data)} vs {len(response)}"
    
    if benchmark_name != "wikirag":
        rag_data = [data_by_id[i] for i in task_index_map["rag"]]
        hallucination_data = [data_by_id[i] for i in task_index_map["hallucination"]]
        rag_response = [response_by_id[i] for i in task_index_map["rag"]]
        hallucination_response = [
            response_by_id[i] for i in task_index_map["hallucination"]
        ]

    else:
        rag_data = [data_by_id[i] for i in task_index_map[1] + task_index_map[2]]
        hallucination_data = [data_by_id[i] for i in task_index_map[3] + task_index_map[4]]
        rag_response = [response_by_id[i] for i in task_index_map[1] + task_index_map[2]]
        hallucination_response = [
            response_by_id[i] for i in task_index_map[3] + task_index_map[4]
        ]

    judgments = []
    if rag_data:
        judgments += llm_judge_rag(
            judger,
            rag_data,
            model_name,
            rag_response,
            0,
            len(data),
            "rag",
            scale=rag_scale,
        )

    if hallucination_data:
        judgments += llm_judge_hallucination(
            judger,
            hallucination_data,
            model_name,
            hallucination_response,
            0,
            len(data),
            "hallucination"
        )

    os.makedirs(f"model_judgement/{benchmark_name}", exist_ok=True)
    append_to_jsonl(f"model_judgement/{benchmark_name}/{judger}.jsonl", judgments)
    result = aggregate_score(judgments, judger)
    samples = aggregate_samples(judgments)
    return result, samples

def aggregate_samples(judgements, is_sorted=True):
    # model = judgements[0]["model"]
    task_data = {}
    for entry in judgements:
        task = entry["task"]
        if task == "rag" or task in [1, 2]:
            entry['avg'] = round((entry["relevance"] + entry["faithfulness"] + entry["completeness"] + entry["utilization"]) / 4, 4)
            if task in task_data:
                task_data[task].append(entry)
            else:
                task_data[task] = [entry]
        elif task == "hallucination" or task in [3, 4]:
            entry['avg'] = round((entry["relevance"] + entry["clarity"] + entry["logical_reasoning"]) / 3, 4)
            if task in task_data:
                task_data[task].append(entry)
            else:
                task_data[task] = [entry]
        else:
            print("Unknown task")
    if is_sorted:
        for task, entries in task_data.items():
            task_data[task] = sorted(entries, key=lambda x: x['avg'])
    return task_data

def aggregate_score(judgements, judger="gpt-4o"):
    model = judgements[0]["model"]
    task_data = {}
    for entry in judgements:
        task = entry["task"]
        if task == "rag" or task in [1, 2]:
            if task not in task_data:
                task_data[task] = {
                    "relevance": 0,
                    "faithfulness": 0,
                    "completeness": 0,
                    "utilization": 0,
                    "count": 0,
                }
            task_data[task]["relevance"] += entry["relevance"]
            task_data[task]["faithfulness"] += entry["faithfulness"]
            task_data[task]["completeness"] += entry["completeness"]
            task_data[task]["utilization"] += entry["utilization"]
            task_data[task]["count"] += 1
        elif task == "hallucination" or task in [3, 4]:
            if task not in task_data:
                task_data[task] = {
                    "relevance": 0,
                    "clarity": 0,
                    "logical_reasoning": 0,
                    "count": 0,
                }
            task_data[task]["relevance"] += entry["relevance"]
            task_data[task]["clarity"] += entry["clarity"]
            task_data[task]["logical_reasoning"] += entry["logical_reasoning"]
            task_data[task]["count"] += 1
        else:
            print("Unknown task")

    aggregate_score = {"model": model, "judger": judger}
    for task, data in task_data.items():
        if task in ["rag", 1, 2]:
            relevance = data["relevance"] / data["count"]
            faithfulness = data["faithfulness"] / data["count"]
            completeness = data["completeness"] / data["count"]
            utilization = data["utilization"] / data["count"]
            if task == "rag":
                aggregate_score['relevance'] = relevance
                aggregate_score['faithfulness'] = faithfulness
                aggregate_score['completeness'] = completeness
                aggregate_score['utilization'] = utilization
                aggregate_score[f"total_{task}_score"] = round(
                    (relevance + faithfulness + completeness + utilization), 4
                )
            else:
                aggregate_score[f"task {task}"] = round(
                    (relevance + faithfulness + completeness + utilization) / 4, 4
                )
        elif task in ["hallucination", 3, 4]:
            relevance = data["relevance"] / data["count"]
            clarity = data["clarity"] / data["count"]
            logical_reasoning = data["logical_reasoning"] / data["count"]
            if task == "hallucination":
                aggregate_score['relevance'] = relevance
                aggregate_score['clarity'] = clarity
                aggregate_score['logical_reasoning'] = logical_reasoning
                aggregate_score[f"total_{task}_score"] = round(
                    (relevance + clarity + logical_reasoning), 4
                )
            else:
                aggregate_score[f"task {task}"] = round(
                    (relevance + clarity + logical_reasoning) / 3, 4
                )
        else:
            print("Unknown task")      
    return aggregate_score


# Main function to parse arguments
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple models based on provided model_list"
    )
    parser.add_argument(
        "--benchmark_name",
        default="wikirag",
        help="Name of the benchmark file",
    )
    parser.add_argument(
        "--model_list", required=True
    )
    parser.add_argument(
        "--judger",
        help="Judge model name, supporting OpenAI, Gemini, and Claude models",
        default="gpt-4o",
        # choices=[
        #     "gpt-4o", "chatgpt-4o-latest", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o1-preview", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
        #     "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b","gemini-1.5-pro",
        #     "claude-3-7-sonnet-latest",  "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022",  "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620",
        #     "claude-3-opus-20240229", "claude-3-opus-latest", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
        # ]
    )
    parser.add_argument(
        "--judge-models",
        help="Judge model name, supporting OpenAI, Gemini, and Claude models",
        default="gpt-4o-mini,gpt-4o",   
    )
    parser.add_argument(
        "--auto-only",
        help="Only auto evaluate the model",
        action="store_true",
    )
    parser.add_argument(
        "--enable-auto-metrics",
        help="Only auto evaluate the model",
        action="store_true",
    )
    parser.add_argument(
        "--rag-scale",
        help="Specify the scale for RAG evaluation",
        type=int,
        default=5,
    )


    args = parser.parse_args()
    benchmark = args.benchmark_name
    rag_scale = args.rag_scale
    model = args.model_list

    global_results_fp = os.environ.get(
        "GLOBAL_RESULTS_FOLDER_PATH", f"results/{get_model_name(model)}"
    )
    Path(global_results_fp).mkdir(parents=True, exist_ok=True)

    if args.enable_auto_metrics:
        auto_metrics_result = auto_evaluate(args.model_list, benchmark)
        if auto_metrics_result is not None:
            f1_score = auto_metrics_result['f1_score'].mean()
            bert_score = auto_metrics_result['bert_score'].mean()
            with open(os.path.join(global_results_fp, f"{benchmark}.jsonl"), "a") as f:
                json.dump({"model": f"{args.model_list}", "f1_score": f1_score, "bert_score": bert_score}, f, ensure_ascii=False)
                f.write("\n")

        if args.auto_only:
            return

    # grouped_sampless = []
    judgers = args.judge_models.split(",")
    print(judgers)
    samples_folder_path = os.path.join(global_results_fp, f"{benchmark}-samples")
    for judger in judgers:
        # Calling the evaluation function with the provided model_list
        result, grouped_samples = evaluate_models(judger, args.model_list, benchmark, rag_scale)
        with open(os.path.join(global_results_fp, f"{benchmark}.jsonl"), "a") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
        samples_judge_folder_path = os.path.join(samples_folder_path, f"judge-{judger}")
        if rag_scale == 2:
            samples_judge_folder_path += "-scale-2"
        os.makedirs(samples_judge_folder_path, exist_ok=True)
        for key, value in grouped_samples.items():
            with open(os.path.join(samples_judge_folder_path, f"task-{key}-samples.jsonl"), "w") as f:
                f.write('\n'.join([json.dumps(sample, ensure_ascii=False) for sample in value]))


if __name__ == "__main__":
    main()

# python gen_judgement.py --model_list little_lora_4 --judge-models DeepSeek-R1 --benchmark_name manual_level_2