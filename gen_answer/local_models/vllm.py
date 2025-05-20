from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def gen_answer(inputs, data, model_path, tensor_parallel_size=4):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.01, top_p=0.01, max_tokens=4096)
    print(
        f"Using model: {model_path} with tensor parallel size: {tensor_parallel_size}"
    )
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )

    texts = []
    for item in inputs:
        messages = [{"role": "user", "content": item["input"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    outputs = llm.generate(texts, sampling_params)

    result_eval = []
    for idx in range(len(outputs)):
        result_eval.append(
            {
                "task": data[idx]["task"],
                "id": data[idx]["id"],
                "response": outputs[idx].outputs[0].text,
                "prompt": data[idx]["input"]
            }
        )

    return result_eval
