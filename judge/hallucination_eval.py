from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm
from judgers.openai import hallu_eval as hallu_eval_openai
from judgers.claude import hallu_eval as hallu_eval_claude
import os
import json

class EvalHallucination(BaseModel):
    relevance: int
    clarity: int
    logical_reasoning: int


def llm_judge(judger, data, model_name, response, start_idx, end_idx, task_label):
    if "gpt" in judger or "o1" in judger or "o3" in judger:
        print("Judging with gpt-4o")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        eval_func = hallu_eval_openai
    elif "gemini" in judger:
        print("Judging with gemini-2.0-flash")
        if not os.getenv("GEMINI_API_KEY"):
            api_key = os.getenv("GOOGLE_AIS_API_KEY")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        judger = "gemini-2.0-flash"
        eval_func = hallu_eval_openai
    elif "claude" in judger:
        print("Judging with claude-3-5-sonnet-latest")
        client = ChatAnthropic(
            model_name="claude-3-5-sonnet-latest",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0001,
            max_tokens=2048,
            top_p=1)
        eval_func = hallu_eval_claude
    
    result_eval = []
    
    for idx, item in tqdm(
        enumerate(data[start_idx:end_idx]),
        desc=f"Evaluating model {model_name} for {task_label}",
    ):
        item_result = eval_func(client, judger, item, model_name, response, idx, start_idx)
        item_result["context"] = item["context"]
        item_result["question"] = item["question"]
        # item_result["label"] = item["output"]
        item_result["prediction"] = response[idx + start_idx]["response"]
        result_eval.append(item_result)
    return result_eval
