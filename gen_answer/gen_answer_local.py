import os
from utils import read_json, write_json, generate_inputs, load_prompts, get_model_name
from local_models.vllm import gen_answer as gen_answer_vllm
from local_models.huggingface import gen_answer as gen_answer_huggingface


def gen_answer_local(model_path, benchmark_name, tensor_parallel_size=1):
    instructions = load_prompts("prompts_fix.txt")
    print("Num instructions:", len(instructions))

    data = read_json(f"../data/{benchmark_name}.json")
    inputs = generate_inputs(data, instructions)
    print(inputs[0]) # check sample input

    model_name = get_model_name(model_path=model_path)

    result_eval = gen_answer_vllm(inputs, data, model_path, tensor_parallel_size)
    # result_eval = gen_answer_huggingface(inputs, data, model_path)

    os.makedirs(f"model_answer/{benchmark_name}", exist_ok=True)
    write_json(result_eval, f"model_answer/{benchmark_name}/{model_name}.json")
    return result_eval
