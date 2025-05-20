from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from prompt_judge import prompt_1_a, prompt_2_a
from prompt_judge_2 import (
    prompt_1_a as prompt_1_a_scale_2
)

class EvalRAG(BaseModel):
    relevance: int
    faithfulness: int
    completeness: int
    utilization: int


class EvalHallucination(BaseModel):
    relevance: int
    clarity: int
    logical_reasoning: int


def rag_eval(
    model: ChatAnthropic,
    judger,
    item,
    model_name,
    response,
    idx,
    start_idx,
    scale=5,
):
    if scale == 2:
        prompt = prompt_1_a_scale_2
    else:
        prompt = prompt_1_a
    
    structured_llm = model.with_structured_output(EvalRAG)
    
    message = structured_llm.invoke(prompt(
        item["context"],
        item["question"],
        item["output"],
        response[idx + start_idx]["response"],
    ))
    
    if message:
        result_eval = {
            "model": model_name,
            "task": item["task"],
            "id": item["id"],
            "relevance": message.relevance,
            "faithfulness": message.faithfulness,
            "completeness": message.completeness,
            "utilization": message.utilization,
        }

        print(
            f' id: {item["id"]}, relevance: {message.relevance}, faithfulness: {message.faithfulness}, completeness: {message.completeness}, utilization: {message.utilization}'
        )

    else:
        print(f' id: {item["id"]}, refusal: {message.refusal}')

    return result_eval


def hallu_eval(
    model: ChatAnthropic,
    judger,
    item,
    model_name,
    response,
    idx,
    start_idx,
):
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
