import os
from utils import read_json, write_json, load_prompts, generate_inputs
from api_models.openai import (
    OPENAI_BASE_URL,
    OPENAI_MODELS,
    gen_answer as gen_answer_openai,
)
from api_models.gemini import GEMINI_BASE_URL, GEMINI_MODELS
from api_models.claude import CLAUDE_MODELS, ANTHROPIC_BASE_URL


def gen_answer_api(
    model_name, benchmark_name, base_url=None, api_key=None, temperature=0.0001
):
    instructions = load_prompts("prompts_fix.txt")
    print(len(instructions))
    num_workers = 20
    data = read_json(f"../data/{benchmark_name}.json")
    inputs = generate_inputs(data, instructions)
    print(inputs[0]) # check sample input
    result_eval = []
    

    if not base_url:
        if model_name in GEMINI_MODELS:
            base_url = GEMINI_BASE_URL
        elif model_name in OPENAI_MODELS:
            base_url = OPENAI_BASE_URL
        elif model_name in CLAUDE_MODELS:
            base_url = ANTHROPIC_BASE_URL
            num_workers = 2

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    # for openai API compatible models
    result_eval = gen_answer_openai(
        inputs, result_eval, model_name, base_url, api_key, temperature, num_workers
    )

    output_filename = f"model_answer/{benchmark_name}/{model_name}.json"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    write_json(result_eval, output_filename)
    return result_eval
