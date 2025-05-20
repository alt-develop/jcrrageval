import re
import os
import json
import retry
from pydantic import BaseModel, Field
from openai import OpenAI
from prompt_judge import prompt_1_a, prompt_2_a, prompt_1_a_r1
from prompt_judge_2 import (
    prompt_1_a as prompt_1_a_scale_2,
    prompt_1_a_r1 as prompt_1_a_r1_scale_2,
)
import logging
from bespokelabs import curator
import datasets
import pandas as pd

logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)

class EvalRAG(BaseModel):
    relevance: int
    faithfulness: int
    completeness: int
    utilization: int


class EvalHallucination(BaseModel):
    relevance: int
    clarity: int
    logical_reasoning: int


def parse_json(evaluation):
    json_regex_pattern = r'```json(.*)```'
    match_object = re.match(pattern=json_regex_pattern, string=evaluation, flags=re.DOTALL)
    
    json_data = match_object.group(1).strip()
    json_data = json.loads(json_data)
    
    return json_data


class RAGJudger(curator.LLM):
    response_format = EvalRAG
    
    def prompt(self, input: dict):
        # print(f"Prompt: {input['prompt']}")
        return input['prompt']

    def parse(self, input: dict, response: EvalRAG):
        return {
            "task": input["task"],
            "id": input["id"],
            "relevance": response.relevance,
            "faithfulness": response.faithfulness,
            "completeness": response.completeness,
            "utilization": response.utilization,
        }
    


@retry.retry(tries=20, delay=1, logger=logging.getLogger(__name__))
def rag_eval_reasoning(
    model: OpenAI, judger, item, model_name, response, idx, start_idx, scale=5
):
    if scale == 2:
        prompt = prompt_1_a_r1_scale_2
    else:
        prompt = prompt_1_a_r1

    completion = model.chat.completions.create(
        model=judger,
        messages=[
            {
                "role": "user",
                "content": prompt(
                    item["context"],
                    item["question"], 
                    item["output"],
                    response[idx + start_idx]["response"],
                )
            }
        ],
    )
    evaluation = completion.choices[0].message.content
    evaluation_json = parse_json(evaluation)

    result_eval = {
        "model": model_name,
        "task": item["task"],
        "id": item["id"],
        "relevance": evaluation_json["relevance"],
        "faithfulness": evaluation_json["faithfulness"],
        "completeness": evaluation_json["completeness"],
        "utilization": evaluation_json["utilization"],
    }
    result_eval['faithfulness'] = min(result_eval['faithfulness'], scale)
    result_eval['relevance'] = min(result_eval['relevance'], result_eval['faithfulness'])
    result_eval['completeness'] = min(result_eval['completeness'], result_eval['faithfulness'])
    result_eval['utilization'] = min(result_eval['utilization'], result_eval['faithfulness'])

    print(
        f'Eval reasoning: id: {item["id"]}, relevance: {result_eval["relevance"]}, faithfulness: {result_eval["faithfulness"]}, completeness: {result_eval["completeness"]}, utilization: {result_eval["utilization"]}'
    )

    return result_eval


def rag_eval(
    client: OpenAI,
    judger,
    data,
    model_name,
    response,
    start_idx,
    scale=5
):
    # print(f"Judging with {judger}")
    # time.sleep()
    if "o1" in judger or "o3" in judger or "o4" in judger:
        if scale == 2:
            prompt = prompt_1_a_r1_scale_2
        else:
            prompt = prompt_1_a_r1
        temperature = 1.0
    else:
        temperature = 0.0001
        if scale == 2:
            prompt = prompt_1_a_scale_2
        else:
            prompt = prompt_1_a

    judge_data = []
    for idx, item in enumerate(data):
        judge_data.append(
            {
                "id": item["id"],
                "task": item["task"],
                "prompt": prompt(
                    item["context"],
                    item["question"],
                    item["output"],
                    response[idx + start_idx]["response"],
                ),
            }
        )
        
    judge_df = pd.DataFrame(judge_data)
    judge_df['id'] = judge_df['id'].astype(str)
    judge_data = datasets.Dataset.from_pandas(judge_df)
    
    # api_key = client.api_key
    # if not api_key:
    #     raise ValueError("API key is required for OpenAI API.")
    # else:
    #     print(api_key[:10] + "..." + api_key[-10:])
    
    judger = RAGJudger(
        model_name=judger,
        # batch=True,
        backend="openai",
        response_format=EvalRAG,
        generation_params={
            "temperature": temperature,
            "max_tokens": 6000,
            "top_p": 1,
        },
        backend_params={
            "api_key": client.api_key,
            "base_url": str(client.base_url),
        }
    )
    result_eval = judger(judge_data, working_dir=".cache/curator")
    result_eval = result_eval.to_pandas()
    
    result_eval['model'] = model_name
    result_eval['faithfulness'] = result_eval['faithfulness'].clip(upper=scale)
    result_eval['relevance'] = result_eval['relevance'].clip(upper=result_eval['faithfulness'])
    result_eval['completeness'] = result_eval['completeness'].clip(upper=result_eval['faithfulness'])
    result_eval['utilization'] = result_eval['utilization'].clip(upper=result_eval['faithfulness'])
    
    return result_eval.to_dict(orient="records")


def hallu_eval(
    client: OpenAI,
    judger,
    item,
    model_name,
    response,
    idx,
    start_idx,
):
    completion = client.beta.chat.completions.parse(
        model=judger,
        messages=[
            {
                "role": "user",
                "content": prompt_2_a(
                    item["context"],
                    item["question"],
                    response[idx + start_idx]["response"],
                ),
            }
        ],
        temperature=0.0001,
        max_tokens=2048,
        top_p=1,
        response_format=EvalHallucination,
    )
    message = completion.choices[0].message
    if message.parsed:
        # Adjust keys if needed for different tasks
        result_eval = {
            "model": model_name,
            "task": item["task"],
            "id": item["id"],
            "relevance": message.parsed.relevance,
            "clarity": message.parsed.clarity,
            "logical_reasoning": message.parsed.logical_reasoning,
        }
        print(f' id: {item["id"]}, relevance: {message.parsed.relevance}, clarity: {message.parsed.clarity}, logical_reasoning: {message.parsed.logical_reasoning}')

    else:
        print(f' id: {item["id"]}, refusal: {message.refusal}')

    return result_eval
