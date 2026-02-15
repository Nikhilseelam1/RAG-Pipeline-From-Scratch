# rag/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llm_model():
    model_id = "google/gemma-2b-it"
    use_quantization_config = False
    attn_implementation = "sdpa"

    print(f"[INFO] Using model_id: {model_id}")
    print(f"[INFO] Using attention implementation: {attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    print("Model loaded successfully")

    return tokenizer, llm_model
