import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Point at your local clone so it never re-downloads
MODEL_NAME = "meta-llama/Llama-3.2-1B"

def load_quantized_model():
    # Load tokenizer from local dir
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the full‑precision model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Apply dynamic quantization to all Linear layers
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return tokenizer, model

def generate_sample(tokenizer, model):
    prompt = "Human: Hello, who are you?\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,        # ← must be True for sampling
        temperature=0.7,       # randomness
        top_p=0.9,             # nucleus sampling
        pad_token_id=tokenizer.eos_token_id
    )

    print("\n=== Response ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    print("Loading and quantizing model…")
    tokenizer, model = load_quantized_model()
    print("Model ready; generating a sample:")
    generate_sample(tokenizer, model)
