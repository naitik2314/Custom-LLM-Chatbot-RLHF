import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1b"  # or ./models/llama-3-1b if downloaded locally

def load_quantized_model():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the full-precision model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Apply dynamic quantization to ALL Linear layers
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return tokenizer, model

def generate_sample(tokenizer, model):
    prompt = "Human: Hello, who are you?\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate with a small max_new_tokens for speed
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    print("\n=== Response ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    print("Loading and quantizing model… this may take 1–2 minutes.")
    tokenizer, model = load_quantized_model()
    print("Model loaded. Generating sample response:")
    generate_sample(tokenizer, model)
