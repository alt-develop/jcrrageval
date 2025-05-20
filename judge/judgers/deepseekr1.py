from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from prompt_judge import prompt_1_a, prompt_2_a, prompt_1_a_r1
import re
import json
import retry
import requests

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

@retry.retry(tries=5, delay=5)
def run_rag_eval(model: OpenAI,
    judger,
    item,
    model_name,
    response,
    idx,
    start_idx,):
    url = 'http://10.2.201.69:8000/v1/chat/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt_1_a_r1(
                    item["context"],
                    item["question"], 
                    item["output"],
                    response[idx + start_idx]["response"],
                ),
                "name": "string"
            }
        ],
        "model": "/data/lhtm-opt3/hub/DeepSeek-R1"
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=3600)
    completion = response.json()
    evaluation = completion["choices"][0]["message"]["content"]
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

    print(
        f' id: {item["id"]}, relevance: {evaluation_json["relevance"]}, faithfulness: {evaluation_json["faithfulness"]}, completeness: {evaluation_json["completeness"]}, utilization: {evaluation_json["utilization"]}'
    )

    return result_eval


def rag_eval(
    model: OpenAI,
    judger,
    item,
    model_name,
    response,
    idx,
    start_idx,
):
    try:
        return run_rag_eval(model, judger, item, model_name, response, idx, start_idx)
    except Exception as e:
        return {
            "model": model_name,
            "task": item["task"],
            "id": item["id"],
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "utilization": 0,
            "error": str(e),
        }


def hallu_eval(
    model: ChatAnthropic,
    judger,
    item,
    model_name,
    response,
    idx,
    start_idx,
):
    print("THIS FUNCTION IS NOT COMPLETED")
    return
    structured_llm = model.with_structured_output(EvalHallucination)
    
    message = structured_llm.invoke(prompt_2_a(
        item["context"],
        item["question"],
        response[idx + start_idx]["response"],
    ))
    
    if message:
        # Adjust keys if needed for different tasks
        result_eval = {
            "model": model_name,
            "task": item["task"],
            "id": item["id"],
            "relevance": message.relevance,
            "clarity": message.clarity,
            "logical_reasoning": message.logical_reasoning,
        }
        print(f' id: {item["id"]}, relevance: {message.relevance}, clarity: {message.clarity}, logical_reasoning: {message.logical_reasoning}')

    else:
        print(f' id: {item["id"]}, refusal: {message.refusal}')

    return result_eval
