import anthropic
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"

CLAUDE_MODELS = [
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


def generate_response(item, model_name, client: anthropic.Anthropic, temperature):
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        messages=[
            {"role": "user", "content": item["input"]}
        ],
        temperature=temperature,
    )
    
    answer = response.content[0]['text']
    # print(f"Answer for {item['id']}: {answer}")
    return {
        "task": item["task"],
        "id": item["id"],
        "response": answer,
        "prompt": item["input"],
    }


def gen_answer(inputs, result_eval, model_name, api_key=None, temperature=0.0001):
    client = anthropic.Anthropic(api_key=api_key)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_response, item, model_name, client, temperature) for item in inputs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating answers for {model_name}"):
            result_eval.append(future.result())
    
    return result_eval
