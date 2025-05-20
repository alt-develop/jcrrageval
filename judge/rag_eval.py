from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from auto_metrics import f1_score, bert_score
import pandas as pd
from judgers.openai import rag_eval as rag_eval_openai
from judgers.claude import rag_eval as rag_eval_claude
from judgers.deepseekr1 import rag_eval as rag_eval_r1
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_item(eval_func, client, judger, item, model_name, response, idx, start_idx, scale=5):
    item_result = eval_func(client, judger, item, model_name, response, idx, start_idx, scale)
    item_result["context"] = item["context"]
    item_result["question"] = item["question"]
    item_result["label"] = item["output"]
    item_result["prediction"] = response[idx + start_idx]["response"]
    return item_result

def llm_judge(judger, data, model_name, response, start_idx, end_idx, task_label, scale=5):
    num_workers = 20
    if "gpt" in judger or "o1" in judger or "o3" in judger or "o4" in judger:
        print(f"Judging with {judger}")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # judger = "gpt-4o"
        eval_func = rag_eval_openai
    elif "gemini" in judger:
        print(f"Judging with {judger}")
        if not os.getenv("GEMINI_API_KEY"):
            api_key = os.getenv("GOOGLE_AIS_API_KEY")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        eval_func = rag_eval_openai
        num_workers = 10
    elif "claude" in judger:
        print(f"Judging with {judger}")
        client = ChatAnthropic(
            model_name=judger,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0001,
            max_tokens=2048,
            top_p=1)
        eval_func = rag_eval_claude
    elif 'r1' in judger.lower():
        judger = '/data/lhtm-opt3/hub/DeepSeek-R1'
        client = OpenAI(
            # api_key=os.getenv("OPENAI_API_KEY"),
            base_url="http://10.2.201.69:8000/v1/chat/completions",
        )
        eval_func = rag_eval_r1
    
    result_eval = rag_eval_openai(client, judger, data[start_idx:end_idx], model_name, response, start_idx, scale)
    

    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [
    #         executor.submit(
    #             process_item, eval_func, client, judger, item, model_name, response, idx, start_idx, scale
    #         )
    #         for idx, item in enumerate(data[start_idx:end_idx])
    #     ]
    #     for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating model {model_name} for {task_label}"):
    #         result_eval.append(future.result())

    return result_eval


def auto_metrics(data, response) -> pd.DataFrame:
    """Evaluate with F1 metrics"""
    df = pd.DataFrame(data)
    response_df = pd.DataFrame(response)

    # sort by id
    df = df.sort_values("id").reset_index(drop=True)
    response_df = response_df.sort_values("id").reset_index(drop=True)

    # check length and same ids
    assert len(df) == len(response_df), "Length mismatch"
    assert df["id"].equals(response_df["id"]), "ID mismatch"

    df["response"] = response_df["response"]
    df["f1_score"] = df.apply(
        lambda x: f1_score(x["output"], x["response"]), axis=1,
    )
    df["bert_score"] = df.apply(
        lambda x: bert_score(x["output"], x["response"]), axis=1,
    )

    print(f"F1 Score: {df['f1_score'].mean()}")
    print(f"BERT Score: {df['bert_score'].mean()}")

    return df[["id", "f1_score", "bert_score"]]
