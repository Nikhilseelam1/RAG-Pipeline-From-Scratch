

def prompt_formatter(query, context_items):
    context_text = "\n\n".join(
        [item["sentence_chunk"] for item in context_items]
    )

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context_text}

    Question:
    {query}

    Answer:
    """
    return prompt


def generate_answer(
    query,
    context_items,
    tokenizer,
    llm_model,
    max_new_tokens,
    temperature
):
    prompt = prompt_formatter(query, context_items)

    input_ids = tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens
    )

    generated_tokens = outputs[0][input_ids["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer
