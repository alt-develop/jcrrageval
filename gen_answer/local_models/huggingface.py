from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from vllm import LLM, SamplingParams


def gen_answer(inputs, data, model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    
    model_kwargs = {"torch_dtype": 'auto', "trust_remote_code": True, "device_map": "auto"}
    tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        temperature=0.1,
        # top_p=0.1,
        **model_kwargs
    )
    
    print(
        f"Using model: {model_path} on HF"
    )
    
    messagess = []
    for item in inputs:
        messages = [{"role": "user", "content": item["input"]}]
        messagess.append(messages)

    
    outputs = llm(messagess, batch_size=16)

    result_eval = []
    for idx in range(len(outputs)):
        result_eval.append(
            {
                "task": data[idx]["task"],
                "id": data[idx]["id"],
                "response": outputs[idx][0]["generated_text"][1]['content'],
                "prompt": data[idx]["input"]
            }
        )

    return result_eval
