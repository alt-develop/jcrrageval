from tqdm import tqdm
import pandas as pd
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from bespokelabs import curator
import datasets

logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)

OPENAI_BASE_URL = "https://api.openai.com/v1"

OPENAI_MODELS = [
    "gpt-4o",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o3-mini",
    "o1-preview",
    "gpt-4o-realtime-preview",
    "gpt-4o-mini-realtime-preview",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct"
]

REASONING_MODELS = [
    "o1",
    "o1-preview",
    "o1-pro",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini"
]


class RAGReponseGenerator(curator.LLM):
    def prompt(self, input: dict):
        return input['input']
    
    def parse(self, input: dict, response: str):
        return {"task": input["task"], "id": input["id"], "response": response, "prompt": input["input"]}


def generate_response(item, model_name, client, temperature):
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": item["input"]}],
        temperature=temperature,
    )
    answer = response.choices[0].message.content
    return {"task": item["task"], "id": item["id"], "response": answer, "prompt": item["input"]}
    
def gen_answer(
    inputs, result_eval, model_name, base_url=None, api_key=None, temperature=0.0001, num_workers=20
):
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
    # if model_name in REASONING_MODELS:
        temperature = 1.0
        
    generator = RAGReponseGenerator(
        model_name=model_name,
        batch=True,
        generation_params={
            "temperature": temperature,
        },
        backend_params={
            "base_url": base_url,
            "api_key": api_key
        }
    )
    
    rag_df = pd.DataFrame(inputs)
    print(rag_df)
    rag_dataset = datasets.Dataset.from_pandas(rag_df)
    
    rag_response_dataset = generator(rag_dataset, working_dir="../.curator")
    print(rag_response_dataset)
    
    # convert the dataset to a list of dictionaries
    result_eval_df = rag_response_dataset.to_pandas()
    result_eval = result_eval_df.to_dict(orient="records")
    
    return result_eval